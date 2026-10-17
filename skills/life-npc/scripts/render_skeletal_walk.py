#!/usr/bin/env python3
"""Continuous two-leg IK walk, alpha-layer compositing, and local MP4 encoding.

Usage: python render_skeletal_walk.py walk.json [--dry-run] [--overwrite]
Paths are relative to the JSON. Joint coordinates are in the complete source PNG.
The path x/y is the on-ground position below the pelvis; scale is output/source.
All output frames are separately articulated: no pose replacement, crossfade,
optical flow, or generated inbetween imagery is used.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from resolve_runtime import resolve_tool


def point(value):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (2,) or not np.isfinite(out).all():
        raise ValueError("A joint coordinate must contain two finite numbers")
    return out


def rotation(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def smooth(value):
    u = min(1.0, max(0.0, value))
    return u*u*(3-2*u)


def alpha_asset(path, alpha_override=None):
    image = Image.open(path).convert("RGBA")
    a = np.asarray(image).copy()
    if alpha_override is not None:
        a[:, :, 3] = np.rint(a[:, :, 3]*alpha_override).astype(np.uint8)
    box = Image.fromarray(a[:, :, 3]).getbbox()
    if box is None:
        raise ValueError(f"Empty alpha asset: {path}")
    crop=a[box[1]:box[3], box[0]:box[2]].astype(np.float32)
    crop[:,:,:3]*=crop[:,:,3:4]/255.
    return {"pixels": crop, "box": box,
            "canvas": image.size, "path": str(path)}


def make_leg(config, root):
    source = (root / config["image"]).resolve()
    image = Image.open(source).convert("RGBA")
    w, h = image.size
    hip, knee, ankle, toe = [point(config[k]) for k in ("hip", "knee", "ankle", "toe")]
    upper_vec, lower_vec = knee-hip, ankle-knee
    length1, length2 = np.linalg.norm(upper_vec), np.linalg.norm(lower_vec)
    if min(length1, length2) < 10:
        raise ValueError("Leg bone lengths must be at least 10 source pixels")
    yy, xx = np.mgrid[:h, :w]
    coordinates = np.stack((xx, yy), axis=-1)
    above_knee = np.sum((coordinates-knee)*(upper_vec/length1), axis=-1)
    above_ankle = np.sum((coordinates-ankle)*(lower_vec/length2), axis=-1)
    overlap = float(config.get("joint_overlap", min(length1, length2)*0.11))
    feather = float(config.get("joint_feather", 2))
    # These are skeletal source-region masks, never RGB colour masks.
    def before(distance, threshold):
        return np.clip((threshold+feather-distance)/max(feather, 0.001), 0, 1)
    thigh = before(above_knee, overlap)
    calf = (1-before(above_knee, -overlap))*before(above_ankle, overlap*0.55)
    foot = 1-before(above_ankle, -overlap*0.55)
    sole = point(config.get("sole", [ankle[0], max(ankle[1]+8, Image.fromarray(np.asarray(image)[:, :, 3]).getbbox()[3]-2)]))
    mesh_texture=np.asarray(image).astype(np.float32)
    mesh_texture[:,:,:3]*=mesh_texture[:,:,3:4]/255.
    rows=sorted(set(list(range(0,h,20))+[h-1]+[int(p[1]+d) for p in (knee,ankle) for d in (-50,0,50) if 0<p[1]+d<h]))
    columns=list(range(0,w,100))+[w-1]
    vertices=np.array([[x,y] for y in rows for x in columns],dtype=np.float32)
    triangles=[]
    for j in range(len(rows)-1):
        for i in range(len(columns)-1):
            if not np.any(mesh_texture[rows[j]:rows[j+1]+1,columns[i]:columns[i+1]+1,3]):
                continue
            a=j*len(columns)+i;b=a+1;c=a+len(columns);d=c+1
            triangles.extend([(a,b,c),(b,d,c)])
    texture_scale = float(config.get("texture_scale", 1))
    if texture_scale <= 0:
        raise ValueError("texture_scale must be positive")
    return {"config": config, "name": config.get("name", "leg"),
            "source": str(source), "hip": hip, "knee": knee, "ankle": ankle, "toe": toe,
            "sole": sole, "L1": float(length1)*texture_scale, "L2": float(length2)*texture_scale,
            "texture_scale": texture_scale,
            "mesh_texture":mesh_texture,"mesh_vertices":vertices,"mesh_triangles":triangles,
            "offset": point(config.get("pelvis_offset", [0, 0])),
            "phase": float(config.get("phase", 0)), "depth": float(config.get("depth", 0)),
            "parts": {"thigh": alpha_asset(source, thigh), "calf": alpha_asset(source, calf),
                      "foot": alpha_asset(source, foot)}}


def load_plan(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    width, height = int(cfg.get("width", 1080)), int(cfg.get("height", 1920))
    fps, duration = float(cfg.get("fps", 30)), float(cfg.get("duration", 6))
    if width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0 or duration <= 0:
        raise ValueError("Invalid canvas, FPS, or duration")
    count = round(fps*duration)
    if abs(count-fps*duration) > 1e-7:
        raise ValueError("duration * fps must be an integer")
    bg = Image.open((root / cfg["background"]).resolve()).convert("RGBA")
    if bg.size != (width, height):
        raise ValueError(f"Background size {bg.size} differs from output {(width,height)}")
    upper_cfg = cfg["upper_body"]
    upper = alpha_asset((root / upper_cfg["image"]).resolve())
    legs = sorted([make_leg(leg, root) for leg in cfg["legs"]], key=lambda leg: leg["depth"])
    if len(legs) != 2:
        raise ValueError("Supply exactly two separate alpha leg assets")
    nodes = cfg["path"]
    if len(nodes) < 2 or nodes[0]["time"] != 0 or any(b["time"] <= a["time"] for a,b in zip(nodes,nodes[1:])):
        raise ValueError("path requires at least two nodes, starting at zero, strictly increasing")
    cycle = float(cfg.get("cycle_seconds", 1.25))
    duty = float(cfg.get("stance_fraction", 0.62))
    if cycle <= 0 or not .5 <= duty <= .8:
        raise ValueError("cycle_seconds > 0 and stance_fraction in [0.5,0.8] required")
    leg_height = np.mean([leg["L1"]+leg["L2"] for leg in legs])
    sole_height = np.mean([(leg["sole"][1]-leg["ankle"][1])*leg["texture_scale"] for leg in legs])
    return {"config": cfg, "config_path": str(path), "root": root, "width": width,
            "height": height, "fps": fps, "duration": duration, "count": count,
            "background": bg, "upper": upper, "pelvis_source": point(upper_cfg["pelvis"]),
            "legs": legs, "path": nodes, "cycle": cycle, "duty": duty,
            "hip_height": float(cfg.get("hip_height_source", .94*leg_height+sole_height)),
            "lift": float(cfg.get("foot_lift_source", 30)),
            "output": (root/cfg["output"]).resolve(),
            "debug": (root/cfg.get("debug_dir", "rig-review")).resolve()}


def path_at(plan, t):
    nodes = plan["path"]
    i = min(len(nodes)-2, max(0, bisect_right([node["time"] for node in nodes], t)-1))
    a,b = nodes[i:i+2]
    u = (t-a["time"])/(b["time"]-a["time"])
    return np.array([a[k]+u*(b[k]-a[k]) for k in ("x","y","scale")], dtype=float)


def plant_at(plan, leg, cycle_start):
    # Place each foot ahead of the moving pelvis at touchdown. Its position then
    # remains unchanged for the complete stance interval, despite root travel.
    state = path_at(plan, cycle_start + plan["cycle"]*plan["duty"]*.5)
    direction = point(plan["config"].get("ground_lateral", [1, 0.10]))
    lateral = float(leg["config"].get("ground_lateral_source", leg["offset"][0]))
    return state[:2] + direction*lateral*state[2]


def foot_state(plan, leg, t):
    cycle, duty = plan["cycle"], plan["duty"]
    raw = t/cycle+leg["phase"]
    n, phase = math.floor(raw), raw % 1
    start = (n-leg["phase"])*cycle
    plant = plant_at(plan, leg, start)
    planted = phase < duty
    if planted:
        sole = plant
        u = phase/duty
        # Subtle heel strike and toe release, with a continuously anchored sole.
        roll = -float(plan["config"].get("heel_strike_degrees", 5))*(1-smooth(u/.25))
        roll += float(plan["config"].get("toe_off_degrees", 9))*smooth((u-.75)/.25)
        lift = 0.
    else:
        u = (phase-duty)/(1-duty)
        next_plant = plant_at(plan, leg, start+cycle)
        sole = plant+(next_plant-plant)*smooth(u)
        lift = plan["lift"]*path_at(plan,t)[2]*math.sin(math.pi*u)**1.2
        sole = sole+np.array([0., -lift])
        # Toe-off bends the shoe for clearance, then prepares the next heel strike.
        roll = (float(plan["config"].get("toe_off_degrees",9))*(1-smooth(u))
                -float(plan["config"].get("heel_strike_degrees",5))*smooth(u))
        roll -= float(plan["config"].get("swing_toe_up_degrees",8))*math.sin(math.pi*u)
    return {"phase": phase, "stance": planted, "stance_start": start,
            "plant": plant, "sole": sole, "lift": lift, "roll": roll}


def solve_knee(hip, ankle, l1, l2, bend):
    vector = ankle-hip
    distance = float(np.linalg.norm(vector))
    d = min(l1+l2-1e-5, max(abs(l1-l2)+1e-5,distance))
    direction = vector/max(distance,1e-8)
    along = (l1*l1-l2*l2+d*d)/(2*d)
    altitude = math.sqrt(max(0., l1*l1-along*along))
    perpendicular = np.array([direction[1], -direction[0]])*bend
    knee = hip+direction*along+perpendicular*altitude
    return knee, max(0., distance-(l1+l2))


def get_state(plan, frame):
    t = frame/plan["fps"]
    base = path_at(plan,t)
    phase = 2*math.pi*t/plan["cycle"]
    bob = float(plan["config"].get("bob_source",4))*math.cos(phase*2)*base[2]
    sway = float(plan["config"].get("hip_sway_source",3))*math.sin(phase)*base[2]
    pelvis = base[:2]+np.array([sway, -plan["hip_height"]*base[2]+bob])
    upper_roll = float(plan["config"].get("upper_sway_degrees", .7))*math.sin(phase)
    feet=[foot_state(plan,leg,t) for leg in plan["legs"]]
    ankles=[foot["sole"]-rotation(foot["roll"])@(leg["sole"]-leg["ankle"])*base[2]*leg["texture_scale"]
            for leg,foot in zip(plan["legs"],feet)]
    if plan["config"].get("adaptive_pelvis",True):
        support=[];weights=[];lower_bounds=[]
        for leg,foot,ankle in zip(plan["legs"],feet,ankles):
            x=pelvis[0]+leg["offset"][0]*base[2]
            dx=float(ankle[0]-x)
            l1,l2=leg["L1"]*base[2],leg["L2"]*base[2]
            target_angle=math.radians(float(plan["config"].get("support_knee_degrees",20)))
            desired_reach=math.sqrt(l1*l1+l2*l2+2*l1*l2*math.cos(target_angle))
            lower_bounds.append(float(ankle[1]-math.sqrt(max(1.,(l1+l2-.05)**2-dx*dx))-leg["offset"][1]*base[2]))
            if foot["stance"]:
                support.append(float(ankle[1]-math.sqrt(max(1.,desired_reach**2-dx*dx))-leg["offset"][1]*base[2]))
                weights.append(math.sin(math.pi*foot["phase"]/plan["duty"])**2+.001)
        if support:
            pelvis[1]=max(float(np.average(support,weights=weights)),max(lower_bounds))
    state = {"frame": frame, "time": t, "ground_root":base[:2].tolist(), "scale": float(base[2]),
             "pelvis":pelvis.tolist(), "upper_roll_degrees":upper_roll, "legs":[]}
    for leg,foot,ankle in zip(plan["legs"],feet,ankles):
        hip = pelvis+leg["offset"]*base[2]
        knee, reach_error = solve_knee(hip, ankle, leg["L1"]*base[2], leg["L2"]*base[2],
                                       float(leg["config"].get("knee_bend",1)))
        state["legs"].append({"name":leg["name"], "hip":hip.tolist(),"knee":knee.tolist(),
                              "ankle":ankle.tolist(),"sole":foot["sole"].tolist(),
                              "phase":foot["phase"],"stance":foot["stance"],
                              "stance_start":foot["stance_start"],"plant":foot["plant"].tolist(),
                              "roll_degrees":foot["roll"],"lift_pixels":foot["lift"],
                              "reach_excess_pixels":reach_error})
    return state


def smooth_pelvis_sequence(plan,states):
    """Anticipate reach constraints, avoiding a hard max at foot handovers.

    A soft temporal upper envelope keeps every ankle reachable while spreading
    the small support-height change over neighbouring frames. This is only a
    numeric bone trajectory filter: no raster image is blended or interpolated.
    """
    if not plan["config"].get("smooth_pelvis",True):
        return
    raw=np.array([state["pelvis"][1] for state in states],dtype=float)
    bounds=[]
    for state in states:
        lower=[]
        for leg,pose in zip(plan["legs"],state["legs"]):
            maximum=(leg["L1"]+leg["L2"])*state["scale"]-.1
            dx=pose["ankle"][0]-pose["hip"][0]
            lower.append(pose["ankle"][1]-math.sqrt(max(1.,maximum*maximum-dx*dx))
                         -leg["offset"][1]*state["scale"])
        bounds.append(max(lower))
    sigma=float(plan["config"].get("pelvis_smooth_frames",2.5))
    filtered=cv2.GaussianBlur(raw.reshape(-1,1),(1,0),sigmaX=0,sigmaY=sigma,
                              borderType=cv2.BORDER_REFLECT).ravel()
    bounds=np.asarray(bounds)
    result=[]
    for i in range(len(states)):
        indices=np.arange(max(0,i-15),min(len(states),i+16))
        candidates=np.append(bounds[indices]-.08*(indices-i)**2,filtered[i])
        largest=float(candidates.max())
        # logsumexp is a differentiable max; temperature=1 output pixel.
        result.append(largest+math.log(float(np.exp(candidates-largest).sum())))
    for state,old,new in zip(states,raw,result):
        shift=float(new-old)
        state["pelvis_raw_y"]=float(old)
        state["pelvis_smoothing_offset_y"]=shift
        state["pelvis"][1]=float(new)
        for leg,pose in zip(plan["legs"],state["legs"]):
            pose["hip"][1]+=shift
            knee,excess=solve_knee(point(pose["hip"]),point(pose["ankle"]),
                                    leg["L1"]*state["scale"],leg["L2"]*state["scale"],
                                    float(leg["config"].get("knee_bend",1)))
            pose["knee"]=knee.tolist()
            pose["reach_excess_pixels"]=excess


def affine(source_anchor, dest_anchor, scale, angle):
    matrix = rotation(angle)*scale
    translation = point(dest_anchor)-matrix@point(source_anchor)
    return np.concatenate((matrix,translation[:,None]),axis=1)


def bone_affine(source_a, source_b, dest_a, dest_b, scale):
    sa,sb,da,db = [point(p) for p in (source_a,source_b,dest_a,dest_b)]
    angle = math.degrees(math.atan2(*(db-da)[::-1])-math.atan2(*(sb-sa)[::-1]))
    return affine(sa,da,scale,angle)


def draw_asset(frame, asset, matrix, clipping=False, brightness=1.):
    box = asset["box"]
    matrix = matrix.copy()
    matrix[:,2] += matrix[:,:2]@np.array(box[:2], dtype=float)
    h,w = asset["pixels"].shape[:2]
    corners = np.array([[0,0],[w,0],[0,h],[w,h]], dtype=float)@matrix[:,:2].T+matrix[:,2]
    left,top = np.floor(corners.min(axis=0)).astype(int)-2
    right,bottom = np.ceil(corners.max(axis=0)).astype(int)+2
    if right < 0 or bottom < 0 or left >= frame.width or top >= frame.height:
        return {"off_canvas":True}
    was_clipped = left < 0 or top < 0 or right > frame.width or bottom > frame.height
    if was_clipped and not clipping:
        raise ValueError(f"A rig asset leaves canvas: {asset['path']}: {[int(left),int(top),int(right),int(bottom)]}")
    left,top,right,bottom=max(0,left),max(0,top),min(frame.width,right),min(frame.height,bottom)
    matrix[:,2]-=np.array([left,top])
    warped = cv2.warpAffine(asset["pixels"],matrix,(right-left,bottom-top),
                            flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    # Resampling is premultiplied-alpha, so hidden transparent RGB can never
    # introduce dark or magenta fringes around hair, parasol ribs, or shoes.
    warped[:,:,:3]=np.divide(warped[:,:,:3]*255.,warped[:,:,3:4],
                             out=np.zeros_like(warped[:,:,:3]),where=warped[:,:,3:4]>.001)*brightness
    warped=np.rint(np.clip(warped,0,255)).astype(np.uint8)
    frame.alpha_composite(Image.fromarray(warped,"RGBA"),(int(left),int(top)))
    return {"bounds":[int(left),int(top),int(right),int(bottom)],"clipped":bool(was_clipped)}


def draw_leg_mesh(frame,leg,matrices,brightness=1.):
    vertices=leg["mesh_vertices"]
    transformed=[]
    knee_y,ankle_y=leg["knee"][1],leg["ankle"][1]
    blend=float(leg["config"].get("mesh_joint_blend",45))
    for p in vertices:
        thigh=matrices[0][:,:2]@p+matrices[0][:,2]
        calf=matrices[1][:,:2]@p+matrices[1][:,2]
        foot=matrices[2][:,:2]@p+matrices[2][:,2]
        if p[1]<knee_y+blend:
            w=smooth((p[1]-(knee_y-blend))/(2*blend))
            q=thigh*(1-w)+calf*w
        else:
            w=smooth((p[1]-(ankle_y-blend))/(2*blend))
            q=calf*(1-w)+foot*w
        transformed.append(q)
    transformed=np.asarray(transformed,dtype=np.float32)
    # Bounds use nontransparent texture triangles rather than empty atlas space.
    used=sorted(set(index for triangle in leg["mesh_triangles"] for index in triangle))
    left,top=np.floor(transformed[used].min(axis=0)).astype(int)-2
    right,bottom=np.ceil(transformed[used].max(axis=0)).astype(int)+2
    left,top,right,bottom=max(0,left),max(0,top),min(frame.width,right),min(frame.height,bottom)
    layer=np.zeros((bottom-top,right-left,4),dtype=np.float32)
    for tri in leg["mesh_triangles"]:
        src=vertices[list(tri)]
        dst=transformed[list(tri)]-np.array([left,top],dtype=np.float32)
        x0,y0=np.floor(dst.min(axis=0)).astype(int)-1
        x1,y1=np.ceil(dst.max(axis=0)).astype(int)+1
        x0,y0,x1,y1=max(0,x0),max(0,y0),min(right-left,x1),min(bottom-top,y1)
        if x1<=x0 or y1<=y0:
            continue
        sx0,sy0=np.floor(src.min(axis=0)).astype(int)
        sx1,sy1=np.ceil(src.max(axis=0)).astype(int)+1
        patch=leg["mesh_texture"][sy0:sy1,sx0:sx1]
        target=dst-np.array([x0,y0],dtype=np.float32)
        matrix=cv2.getAffineTransform(src-np.array([sx0,sy0],dtype=np.float32),target)
        warped=cv2.warpAffine(patch,matrix,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        mask=np.zeros((y1-y0,x1-x0),dtype=np.uint8)
        cv2.fillConvexPoly(mask,np.rint(target*16).astype(np.int32),255,lineType=cv2.LINE_8,shift=4)
        roi=layer[y0:y1,x0:x1]
        roi[mask>0]=warped[mask>0]
    layer[:,:,:3]=np.divide(layer[:,:,:3]*255.,layer[:,:,3:4],out=np.zeros_like(layer[:,:,:3]),
                            where=layer[:,:,3:4]>.001)*brightness
    layer=np.rint(np.clip(layer,0,255)).astype(np.uint8)
    frame.alpha_composite(Image.fromarray(layer,"RGBA"),(int(left),int(top)))
    return {"bounds":[int(left),int(top),int(right),int(bottom)],"mesh_triangles":len(leg["mesh_triangles"]),"single_texture":True}


def render_frame(plan,state):
    frame = plan["background"].copy()
    scale = state["scale"]
    # Independent soft ground ellipse, rendered behind both feet.
    shadow = Image.new("RGBA",(250,90))
    ImageDraw.Draw(shadow).ellipse((20,30,230,65),fill=(0,0,0,int(plan["config"].get("shadow_alpha",40))))
    shadow=shadow.filter(ImageFilter.GaussianBlur(11)).resize((max(1,round(250*scale)),max(1,round(90*scale))),Image.Resampling.BILINEAR)
    root = state["ground_root"]
    frame.alpha_composite(shadow,(round(root[0]-shadow.width/2),round(root[1]-shadow.height*.54)))
    clips=[]
    for leg,pose in zip(plan["legs"],state["legs"]):
        texture_scale=scale*leg["texture_scale"]
        thigh = bone_affine(leg["hip"],leg["knee"],pose["hip"],pose["knee"],texture_scale)
        calf = bone_affine(leg["knee"],leg["ankle"],pose["knee"],pose["ankle"],texture_scale)
        shoe = affine(leg["ankle"],pose["ankle"],texture_scale,pose["roll_degrees"])
        clips.append(draw_leg_mesh(frame,leg,(thigh,calf,shoe),float(leg["config"].get("brightness",1))))
    upper_matrix=affine(plan["pelvis_source"],state["pelvis"],scale,state["upper_roll_degrees"])
    clips.append(draw_asset(frame,plan["upper"],upper_matrix,plan["config"].get("allow_canvas_clipping",False)))
    state["render_bounds"] = clips
    return frame.convert("RGB")


def gait_audit(states):
    groups={}
    for state in states:
        for leg in state["legs"]:
            if leg["stance"]:
                groups.setdefault((leg["name"],leg["stance_start"]),[]).append(np.array(leg["sole"]))
    drifts=[float(np.max(np.linalg.norm(np.stack(points)-points[0],axis=1))) for points in groups.values()]
    pelvis_y=np.array([state["pelvis"][1] for state in states])
    foot_delta=np.array([state["legs"][0]["ankle"][0]-state["legs"][1]["ankle"][0] for state in states])
    return {"max_analytic_sole_drift_during_stance_pixels":max(drifts,default=0.),
            "max_ik_reach_excess_pixels":max(leg["reach_excess_pixels"] for state in states for leg in state["legs"]),
            "max_pelvis_y_frame_step_pixels":float(np.abs(np.diff(pelvis_y)).max()),
            "max_pelvis_y_velocity_change_pixels":float(np.abs(np.diff(pelvis_y,2)).max()),
            "far_minus_near_ankle_x_range_pixels":[float(foot_delta.min()),float(foot_delta.max())],
            "both_feet_take_lead":bool(foot_delta.min()<-5 and foot_delta.max()>5),
            "stance_groups":len(groups),"frames_articulated":len(states),
            "no_pose_swaps":True,"no_optical_flow":True,"no_pose_dissolves":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--overwrite",action="store_true",help="Replace an existing output video")
    args=parser.parse_args()
    plan=load_plan(args.config.expanduser().resolve())
    if plan["output"].exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output already exists; choose a new path or pass --overwrite: {plan['output']}")
    states=[get_state(plan,i) for i in range(plan["count"])]
    smooth_pelvis_sequence(plan,states)
    audit=gait_audit(states)
    plan["debug"].mkdir(parents=True,exist_ok=True)
    if audit["max_ik_reach_excess_pixels"]>.1:
        raise ValueError(f"Unreachable leg endpoint; lower hip_height_source: {audit}")
    plan["output"].parent.mkdir(parents=True,exist_ok=True)
    metadata={"width":plan["width"],"height":plan["height"],"fps":plan["fps"],"duration":plan["duration"],
              "frames":plan["count"],"method":"continuous 2-bone IK; world-space stance sole lock; support-driven pelvis; continuous single-texture triangle-strip leg mesh; separately rolled shoe",**audit}
    if args.dry_run:
        # Include full art placement so a successful dry run validates alpha bounds.
        for i in sorted(set([0,plan["count"]-1]+[min(plan["count"]-1,round(i*plan["cycle"]*plan["fps"]/8)) for i in range(8)])):
            render_frame(plan,states[i]).save(plan["debug"]/f"frame-{i:04d}.png")
        (plan["debug"]/"states.json").write_text(json.dumps({"metadata":metadata,"frames":states},ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(metadata,ensure_ascii=False,indent=2))
        return
    ffmpeg=resolve_tool("ffmpeg",args.ffmpeg,plan["root"])
    ffprobe=resolve_tool("ffprobe",args.ffprobe,plan["root"])
    temporary=plan["output"].with_name(plan["output"].stem+".encoding-"+uuid.uuid4().hex+".mp4")
    command=[str(ffmpeg),"-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pixel_format","rgb24",
             "-video_size",f"{plan['width']}x{plan['height']}","-framerate",str(plan["fps"]),"-i","pipe:0","-an",
             "-c:v","libx264","-preset",plan["config"].get("preset","fast"),"-crf",str(plan["config"].get("crf",17)),
             "-pix_fmt","yuv420p","-movflags","+faststart",str(temporary)]
    selected=set([0,plan["count"]-1]+[min(plan["count"]-1,round(i*plan["cycle"]*plan["fps"]/8)) for i in range(8)])
    selected.update(min(plan["count"]-1,max(0,round(t*plan["fps"]))) for t in plan["config"].get("debug_times",[2.5,4]))
    started=time.monotonic()
    process=None
    try:
        with (plan["debug"]/"ffmpeg.log").open("w") as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            for state in states:
                image=render_frame(plan,state)
                if state["frame"] in selected:
                    image.save(plan["debug"]/f"frame-{state['frame']:04d}.png")
                process.stdin.write(np.asarray(image).tobytes())
            process.stdin.close()
            status=process.wait()
            if status:
                raise RuntimeError(f"FFmpeg failed, exit {status}; see {plan['debug']/'ffmpeg.log'}")
        probe=json.loads(subprocess.check_output([str(ffprobe),"-v","error","-count_frames","-select_streams","v:0",
                                                  "-show_streams","-show_format","-of","json",str(temporary)],text=True))
        if args.overwrite:
            temporary.replace(plan["output"])
        else:
            # Publish without replacing a file created by another render meanwhile.
            os.link(temporary,plan["output"])
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
    metadata["render_seconds"]=time.monotonic()-started
    metadata["output"]=str(plan["output"])
    metadata["ffprobe"]=probe
    (plan["debug"]/"states.json").write_text(json.dumps({"metadata":metadata,"frames":states},ensure_ascii=False,indent=2),encoding="utf-8")
    (plan["debug"]/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
