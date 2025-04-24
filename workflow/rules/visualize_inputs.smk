rule plot_renewable_classes:
    input:
        province_shape=DERIVED_COMMON + "/province_shapes/CHN_adm1.shp",
        renewable_classes=DERIVED_CUTOUT
        + "/"
        + "{technology}_regions_by_class_{rc_params}.geojson",
        average_distance=DERIVED_CUTOUT
        + "/"
        + "average_distance_{technology}-{rc_params}.h5",
    output:
        renewable_grades_bins=DERIVED_CUTOUT + "/" + "{technology}_{rc_params}_bins.png",
        renewable_grades_cf=DERIVED_CUTOUT + "/" + "{technology}_{rc_params}_cfs.png",
        distances_hist=DERIVED_CUTOUT
        + "/"
        + "{technology}_{rc_params}_avg_distances.png",
    script:
        "../scripts/plot_inputs_visualisation.py"
