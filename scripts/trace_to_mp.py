import argparse, json, time, math
from pymavlink import mavutil

def median(xs):
    xs = sorted(xs)
    return xs[len(xs)//2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace JSON with hist.* arrays")
    ap.add_argument("--out", default="tcp:127.0.0.1:5771", help="MAVLink target (router input or SITL)")
    ap.add_argument("--hz", type=float, default=20.0, help="stream rate")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--loop", action="store_true", help="loop playback")
    args = ap.parse_args()

    d = json.load(open(args.trace, "r", encoding="utf-8"))
    h = d.get("hist", {})

    t = h.get("t", [])
    if not t:
        raise SystemExit("No hist.t in trace")

    alpha_rms = h.get("alpha_deg_rms", [])
    ft_rms = h.get("ft_tan_rms", [])
    speed = h.get("speed", [])
    dist = h.get("dist_to_obstacle", [])
    fx = h.get("fx_cmd", [])
    fy = h.get("fy_cmd", [])
    fan16 = h.get("fan_thrust_16", [])
    alpha32 = h.get("alpha_deg_32", [])

    m = mavutil.mavlink_connection(args.out)
    print(f"Streaming trace -> {args.out} at {args.hz} Hz")

    dt = 1.0 / args.hz
    i = args.start

    while True:
        if i >= len(t):
            if args.loop:
                i = 0
            else:
                break

        now = time.time()

        def nv(name, val):
            if val is None:
                return
            try:
                v = float(val)
            except Exception:
                return
            m.mav.named_value_float_send(
                int(now), name.encode("ascii")[:10], v
            )

        if i < len(alpha_rms): nv("alpha_rms", alpha_rms[i])
        if i < len(ft_rms):    nv("ft_rms", ft_rms[i])
        if i < len(speed):     nv("speed", speed[i])
        if i < len(dist):      nv("dist", dist[i])
        if i < len(fx):        nv("fx_cmd", fx[i])
        if i < len(fy):        nv("fy_cmd", fy[i])

        if i < len(fan16) and fan16[i]:
            vals = [float(x) for x in fan16[i]]
            mean = sum(vals)/len(vals)
            if mean > 1e-9:
                var = sum((x-mean)**2 for x in vals)/len(vals)
                std = math.sqrt(var)
                maxdev = max(abs(x-mean) for x in vals)
                nv("fan_std%", 100.0*(std/mean))
                nv("fan_max%", 100.0*(maxdev/mean))

        if i < len(alpha32) and alpha32[i]:
            a = alpha32[i]
            picks = [0,4,8,12,16,20,24,28]
            for k, idx in enumerate(picks):
                if idx < len(a):
                    nv(f"a{k}", a[idx])

        elapsed = time.time() - now
        if elapsed < dt:
            time.sleep(dt - elapsed)
        i += 1

    print("Done.")


if __name__ == "__main__":
    main()
