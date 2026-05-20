from surface_gexit_job_common import run_surface_distance


if __name__ == "__main__":
    run_surface_distance(
        distance=7,
        samples=80000,
        seed=70070,
        workers=20,
        merge_existing=True,
        repeat_label="surface7_addon_80000_seed70070",
    )
