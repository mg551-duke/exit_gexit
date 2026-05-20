from surface_gexit_job_common import run_surface_distance


if __name__ == "__main__":
    run_surface_distance(
        distance=11,
        samples=80000,
        seed=110110,
        workers=20,
        merge_existing=True,
        repeat_label="surface11_addon_80000_seed110110",
    )
