import tarfile, io, numpy as np, matplotlib.pyplot as plt, os

flow_tar_path = "/teamspace/studios/this_studio/AR-flares/wds_flow/train/shard-000315.tar"
save_dir = "./flow_examples"
os.makedirs(save_dir, exist_ok=True)

with tarfile.open(flow_tar_path, "r") as tf:
    flow_members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".flow.npz")]
    print(f"Found {len(flow_members)} flow maps.")

    for i, member in enumerate(flow_members[:3]):  # just take first 3
        flow_bytes = tf.extractfile(member).read()
        with np.load(io.BytesIO(flow_bytes)) as f:
            u = f["u8"].astype(np.float32) * float(f["su"])
            v = f["v8"].astype(np.float32) * float(f["sv"])
        mag = np.sqrt(u**2 + v**2)

        # save visuals
        for name, data, cmap in [
            ("u", u, "RdBu"),
            ("v", v, "RdBu"),
            ("mag", mag, "inferno"),
        ]:
            plt.imshow(data, cmap=cmap)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{os.path.basename(member.name)}_{name}.png"), dpi=150, bbox_inches="tight")
            plt.close()

print("Saved flow visualizations to:", save_dir)
