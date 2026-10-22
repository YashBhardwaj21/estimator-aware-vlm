#!/usr/bin/env python3
"""Record a D455 RGB-D dataset until manually stopped; validate it afterward."""
from __future__ import annotations
import argparse, json, queue, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    raise SystemExit("ERROR: pyrealsense2 is unavailable in this Python environment.")

WIDTH, HEIGHT, FPS = 640, 480, 30

class DiskWriter:
    """Asynchronous paired PNG writer that reports failures instead of dropping silently."""
    def __init__(self, max_queue=300):
        self.q = queue.Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self.errors = []
        self.written_pairs = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def submit(self, rgb_path, rgb, depth_path, depth):
        if self.errors:
            raise RuntimeError("PNG writer failed: " + self.errors[0])
        try:
            self.q.put((rgb_path, rgb.copy(), depth_path, depth.copy()), timeout=5)
        except queue.Full as exc:
            raise RuntimeError("PNG writer queue remained full for 5 seconds; stopping to avoid silent data loss") from exc
    def _run(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try: item = self.q.get(timeout=.1)
            except queue.Empty: continue
            try:
                rp, rgb, dp, depth = item
                if not cv2.imwrite(str(rp), rgb): raise OSError(f"Failed writing {rp}")
                if not cv2.imwrite(str(dp), depth): raise OSError(f"Failed writing {dp}")
                self.written_pairs += 1
            except Exception as exc:
                self.errors.append(str(exc)); self.stop_event.set()
            finally: self.q.task_done()
    def close(self):
        self.q.join(); self.stop_event.set(); self.thread.join()
        if self.errors: raise RuntimeError("PNG writer error: " + "; ".join(self.errors))

def get_intrinsics(profile):
    i = profile.get_intrinsics()
    return {"width": int(i.width), "height": int(i.height), "fx": float(i.fx), "fy": float(i.fy),
            "cx": float(i.ppx), "cy": float(i.ppy), "distortion_model": str(i.model),
            "distortion_coefficients": [float(x) for x in i.coeffs]}
def percentile(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else None

def validate(out, n, written, rgb_ts, depth_ts, rgb_nums, depth_nums, depth_scale, fps, width, height):
    errors, warnings = [], []
    rgb_files, depth_files = sorted((out/'rgb').glob('*.png')), sorted((out/'depth').glob('*.png'))
    assoc = [x for x in (out/'associations.txt').read_text().splitlines() if x.strip() and not x.startswith('#')]
    for label, got in [('RGB PNGs', len(rgb_files)), ('depth PNGs', len(depth_files)), ('association rows', len(assoc)), ('written PNG pairs', written)]:
        if got != n: errors.append(f'{label}: expected {n}, found {got}')
    sample_checks=[]
    decode_failures=[]; dimension_failures=[]; valid_fractions=[]; depth_sample_values=[]
    # Decode every saved frame; sample-only checks can miss a corrupt frame in the middle.
    for idx, path in enumerate(rgb_files):
        im=cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            decode_failures.append(f'RGB frame {idx}: decode failed ({path.name})')
        elif im.shape != (height,width,3) or im.dtype != np.uint8:
            dimension_failures.append(f'RGB frame {idx}: shape/type {im.shape}/{im.dtype}')
    for idx, path in enumerate(depth_files):
        im=cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            decode_failures.append(f'depth frame {idx}: decode failed ({path.name})')
        elif im.shape != (height,width) or im.dtype != np.uint16:
            dimension_failures.append(f'depth frame {idx}: shape/type {im.shape}/{im.dtype}')
        else:
            valid=im>0; valid_fractions.append(float(valid.mean()))
            vals=im[valid]
            if vals.size:
                depth_sample_values.extend((vals[::max(1, vals.size//3000)].astype(np.float64)*depth_scale).tolist())
    errors.extend(decode_failures[:20]); errors.extend(dimension_failures[:20])
    if len(decode_failures)>20: errors.append(f'... plus {len(decode_failures)-20} more decode failures')
    if len(dimension_failures)>20: errors.append(f'... plus {len(dimension_failures)-20} more dimension/type failures')
    if rgb_files:
        for idx in sorted(set([0, len(rgb_files)//2, len(rgb_files)-1])):
            r=cv2.imread(str(rgb_files[idx]), cv2.IMREAD_COLOR)
            d=cv2.imread(str(depth_files[idx]), cv2.IMREAD_UNCHANGED) if idx < len(depth_files) else None
            rok=r is not None and r.shape==(height,width,3) and r.dtype==np.uint8
            dok=d is not None and d.ndim==2 and d.dtype==np.uint16 and d.shape==(height,width)
            sample_checks.append({'index':idx,'rgb_shape':None if r is None else list(r.shape),'rgb_dtype':None if r is None else str(r.dtype),'depth_shape':None if d is None else list(d.shape),'depth_dtype':None if d is None else str(d.dtype),'rgb_valid':bool(rok),'depth_valid':bool(dok)})
    rt=np.asarray(rgb_ts,float); dt=np.asarray(depth_ts,float)
    rdiff=np.diff(rt); ddiff=np.diff(dt); skew=(dt-rt)*1000 if len(dt)==len(rt) else np.array([])
    if len(rdiff) and np.any(rdiff<=0): errors.append('RGB timestamps are not strictly increasing')
    if len(ddiff) and np.any(ddiff<=0): errors.append('Depth timestamps are not strictly increasing')
    if len(skew) and np.max(np.abs(skew))>100: warnings.append('At least one RGB/depth timestamp skew exceeds 100 ms')
    rgaps=sum(b != a+1 for a,b in zip(rgb_nums,rgb_nums[1:]))
    dgaps=sum(b != a+1 for a,b in zip(depth_nums,depth_nums[1:]))
    if rgaps: warnings.append(f'{rgaps} nonconsecutive RGB device-frame transitions; inspect for dropped frames')
    if dgaps: warnings.append(f'{dgaps} nonconsecutive depth device-frame transitions; inspect for dropped frames')
    measured=(n-1)/(rt[-1]-rt[0]) if n>1 and rt[-1]>rt[0] else 0.0
    if n>1 and measured < .7*fps: warnings.append(f'Measured device-timestamp FPS {measured:.2f} is below 70% of requested {fps}')
    depth_valid=valid_fractions; depth_values=depth_sample_values
    if depth_valid and float(np.mean(depth_valid))<.25: warnings.append('Mean valid-depth fraction in sampled frames is below 25%; inspect scene/range')
    video_checks={}
    for name in ('color.mp4','preview_side_by_side.mp4'):
        p=out/name; cap=cv2.VideoCapture(str(p)); opened=cap.isOpened(); ok, frame=cap.read() if opened else (False,None); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0; cap.release()
        video_checks[name]={'exists':p.exists(),'opened':opened,'first_frame_readable':bool(ok and frame is not None),'reported_frames':count}
        if not p.exists() or not opened or not ok: errors.append(f'Video validation failed: {name}')
        if opened and count > 0 and abs(count-n) > max(2, int(0.01*n)):
            warnings.append(f'{name} reports {count} frames for {n} captured frame pairs')
    report={'status':'PASS' if not errors else 'FAIL','errors':errors,'warnings':warnings,
      'counts':{'captured_frames':n,'rgb_pngs':len(rgb_files),'depth_pngs':len(depth_files),'association_rows':len(assoc),'written_png_pairs':written},
      'timing':{'requested_fps':fps,'measured_rgb_fps_from_device_timestamps':round(measured,3),'rgb_interval_ms_median':percentile(rdiff*1000,50),'rgb_interval_ms_p95':percentile(rdiff*1000,95),'depth_interval_ms_median':percentile(ddiff*1000,50),'rgb_depth_skew_ms_median':percentile(skew,50),'rgb_depth_skew_abs_ms_p95':percentile(np.abs(skew),95),'rgb_timestamps_increasing':bool(not len(rdiff) or np.all(rdiff>0)),'depth_timestamps_increasing':bool(not len(ddiff) or np.all(ddiff>0))},
      'device_frame_number_gaps':{'rgb':rgaps,'depth':dgaps},
      'depth_quality':{'sampled_frame_count':len(depth_valid),'mean_valid_pixel_fraction':float(np.mean(depth_valid)) if depth_valid else None,'sampled_depth_m_p05':float(np.percentile(depth_values,5)) if depth_values else None,'sampled_depth_m_median':float(np.median(depth_values)) if depth_values else None,'sampled_depth_m_p95':float(np.percentile(depth_values,95)) if depth_values else None},
      'sample_image_checks':sample_checks,'video_checks':video_checks}
    return report

def main():
    ap=argparse.ArgumentParser(description='Record D455 RGB-D until q/Esc/Ctrl+C. No duration limit.')
    ap.add_argument('--name','-n',default=''); ap.add_argument('--output-dir','-o',default='data/raw')
    ap.add_argument('--width',type=int,default=WIDTH); ap.add_argument('--height',type=int,default=HEIGHT); ap.add_argument('--fps',type=int,default=FPS)
    ap.add_argument('--serial',default='',help='Optional exact device serial'); ap.add_argument('--max-depth-m',type=float,default=5.0,help='Depth preview display range only')
    a=ap.parse_args()
    if min(a.width,a.height,a.fps)<=0 or a.max_depth_m<=0: ap.error('width, height, fps and max-depth-m must be positive')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); name=a.name.strip() or f'd455_rgbd_{stamp}'
    out=(Path(a.output_dir).expanduser().resolve()/name)
    if out.exists() and any(out.iterdir()): raise SystemExit(f'ERROR: output directory is not empty: {out}')
    (out/'rgb').mkdir(parents=True,exist_ok=True); (out/'depth').mkdir(parents=True,exist_ok=True)
    pipeline=rs.pipeline(); cfg=rs.config()
    if a.serial: cfg.enable_device(a.serial)
    cfg.enable_stream(rs.stream.color,a.width,a.height,rs.format.bgr8,a.fps)
    cfg.enable_stream(rs.stream.depth,a.width,a.height,rs.format.z16,a.fps)
    try: profile=pipeline.start(cfg)
    except Exception as e: raise SystemExit(f'ERROR starting D455: {e}\nCheck USB 3.x passthrough, permissions, and competing camera apps.')
    color_writer=preview_writer=disk=assoc=rgb_txt=depth_txt=None
    rgb_ts=[]; depth_ts=[]; rgb_nums=[]; depth_nums=[]; n=0; stop_reason='unknown'; t0=time.time()
    try:
        dev=profile.get_device()
        usb=dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else 'unknown'
        info={'name':dev.get_info(rs.camera_info.name),'serial':dev.get_info(rs.camera_info.serial_number),'firmware':dev.get_info(rs.camera_info.firmware_version),'usb_type_descriptor':usb,'sdk_version':getattr(rs,'__version__','unavailable')}
        if str(usb).startswith('2.'): print('WARNING: camera reports USB 2.x; stop and fix connection before research capture.')
        scale=float(dev.first_depth_sensor().get_depth_scale())
        cp=profile.get_stream(rs.stream.color).as_video_stream_profile(); dp=profile.get_stream(rs.stream.depth).as_video_stream_profile()
        meta={'sequence_name':name,'recording_started_utc':datetime.now(timezone.utc).isoformat(),'device':info,'requested_profile':{'width':a.width,'height':a.height,'fps':a.fps,'color_format':'BGR8','depth_format':'Z16','depth_aligned_to':'color'},'depth_scale_m_per_unit':scale,'depth_units_per_meter':1.0/scale if scale else None,'color_intrinsics':get_intrinsics(cp),'depth_intrinsics':get_intrinsics(dp),'notes':['Depth PNG values are raw uint16 device units.','Depth frames are aligned to the color pixel grid.','Timestamps are device timestamps in seconds.','MP4 files are visualization only; use PNGs for depth processing.']}
        try:
            ex=cp.get_extrinsics_to(dp); meta['color_to_depth_extrinsics']={'rotation':list(ex.rotation),'translation_m':list(ex.translation)}
        except Exception as e: meta['color_to_depth_extrinsics']={'unavailable_reason':str(e)}
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        color_writer=cv2.VideoWriter(str(out/'color.mp4'),fourcc,float(a.fps),(a.width,a.height))
        preview_writer=cv2.VideoWriter(str(out/'preview_side_by_side.mp4'),fourcc,float(a.fps),(a.width*2,a.height))
        if not color_writer.isOpened() or not preview_writer.isOpened(): raise RuntimeError('OpenCV MP4 writer failed to open; check codec support/output permissions')
        disk=DiskWriter(); assoc=open(out/'associations.txt','w',buffering=1); rgb_txt=open(out/'rgb.txt','w',buffering=1); depth_txt=open(out/'depth.txt','w',buffering=1)
        assoc.write('# rgb_timestamp rgb_path depth_timestamp depth_path\n'); rgb_txt.write('# timestamp filename\n'); depth_txt.write('# timestamp filename\n')
        align=rs.align(rs.stream.color)
        print('='*72); print('D455 RGB-D RECORDER — NO TIME LIMIT'); print(f'Output: {out}'); print(f"Device: {info['name']} | serial={info['serial']} | firmware={info['firmware']} | USB={usb}"); print(f'SDK={info["sdk_version"]} | depth scale={scale} m/unit | profile={a.width}x{a.height}@{a.fps}'); print('Move slowly through the scene. Press q/Esc in preview or Ctrl+C to stop.'); print('='*72)
        consecutive_errors=0
        while True:
            try:
                frames=pipeline.wait_for_frames(timeout_ms=5000); aligned=align.process(frames); cf=aligned.get_color_frame(); df=aligned.get_depth_frame()
                if not cf or not df:
                    consecutive_errors+=1; print('WARNING: incomplete RGB-D frameset')
                    if consecutive_errors>=5: raise RuntimeError('Five consecutive incomplete RGB-D framesets')
                    continue
                color=np.asanyarray(cf.get_data()); depth=np.asanyarray(df.get_data())
                if color.shape!=(a.height,a.width,3) or color.dtype!=np.uint8 or depth.shape!=(a.height,a.width) or depth.dtype!=np.uint16: raise RuntimeError(f'Unexpected frame format RGB={color.shape}/{color.dtype}, depth={depth.shape}/{depth.dtype}')
                consecutive_errors=0; rts=float(cf.get_timestamp())/1000.; dts=float(df.get_timestamp())/1000.; rn=int(cf.get_frame_number()); dn=int(df.get_frame_number())
                if not np.isfinite(rts) or not np.isfinite(dts): raise RuntimeError('Non-finite device timestamp')
                rp=f'rgb/{n:06d}.png'; dp=f'depth/{n:06d}.png'
                disk.submit(out/rp,color,out/dp,depth)
                assoc.write(f'{rts:.9f} {rp} {dts:.9f} {dp}\n'); rgb_txt.write(f'{rts:.9f} {rp}\n'); depth_txt.write(f'{dts:.9f} {dp}\n')
                color_writer.write(color)
                dm=depth.astype(np.float32)*scale; vis=np.clip(dm/a.max_depth_m*255,0,255).astype(np.uint8); vis[depth==0]=0; dcolor=cv2.applyColorMap(vis,cv2.COLORMAP_JET); dcolor[depth==0]=0
                combined=np.hstack((color,dcolor)); preview_writer.write(combined)
                n+=1; rgb_ts.append(rts); depth_ts.append(dts); rgb_nums.append(rn); depth_nums.append(dn)
                elapsed=time.time()-t0; host_fps=n/elapsed if elapsed>0 else 0; skew=(dts-rts)*1000
                display=combined.copy(); cv2.rectangle(display,(8,8),(a.width*2-8,52),(0,0,0),-1)
                cv2.putText(display,f'REC | {n} frames | {host_fps:.1f} host FPS | skew {skew:+.2f} ms | q/Esc stop',(16,36),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow('D455 RGB-D Recorder — q/Esc to stop',display); key=cv2.waitKey(1)&0xff
                if key in (ord('q'),27): stop_reason='preview_key'; break
                if n%30==0: print(f'\rFrames={n} | host FPS={host_fps:.1f} | depth valid={np.mean(depth>0)*100:.1f}% | skew={skew:+.2f} ms',end='',flush=True)
            except KeyboardInterrupt: stop_reason='keyboard_interrupt'; print('\nCtrl+C received; finalizing'); break
            except Exception as e: print(f'\nCAPTURE ERROR: {e}',file=sys.stderr); stop_reason='capture_error'; break
    finally:
        try: pipeline.stop()
        except Exception: pass
        if color_writer: color_writer.release()
        if preview_writer: preview_writer.release()
        cv2.destroyAllWindows()
        for h in (assoc,rgb_txt,depth_txt):
            if h: h.close()
        written=disk.written_pairs if disk else 0
        try:
            if disk: disk.close(); written=disk.written_pairs
        except Exception as e: print(f'PNG writer finalization error: {e}',file=sys.stderr); stop_reason='disk_writer_error'
        elapsed=max(time.time()-t0,0)
        try: meta=json.loads((out/'metadata.json').read_text())
        except Exception: meta={}
        meta.update({'recording_ended_utc':datetime.now(timezone.utc).isoformat(),'duration_seconds':round(elapsed,3),'captured_frames':n,'disk_writer_saved_pairs':written,'stop_reason':stop_reason,'host_average_fps':round(n/elapsed,3) if elapsed else 0})
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
        try: report=validate(out,n,written,rgb_ts,depth_ts,rgb_nums,depth_nums,float(meta.get('depth_scale_m_per_unit',.001)),a.fps,a.width,a.height)
        except Exception as e: report={'status':'FAIL','errors':[f'Validation itself failed: {e}'],'warnings':[]}
        (out/'validation_report.json').write_text(json.dumps(report,indent=2))
        (out/'validation_report.txt').write_text('Dataset validation: '+report.get('status','UNKNOWN')+'\n\nErrors:\n'+''.join('- '+x+'\n' for x in report.get('errors',[]))+'\nWarnings:\n'+''.join('- '+x+'\n' for x in report.get('warnings',[]))+'\nFull details: validation_report.json\n')
        print('\n'+'='*72); print('RECORDING FINALIZED'); print(f'Output: {out}'); print(f'Captured frames={n}; PNG pairs written={written}'); print('Validation:',report.get('status','UNKNOWN'))
        for x in report.get('errors',[]): print('ERROR:',x)
        for x in report.get('warnings',[]): print('WARNING:',x)
        print(f'Report: {out}/validation_report.txt'); print('='*72)
    return 0 if report.get('status')=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
