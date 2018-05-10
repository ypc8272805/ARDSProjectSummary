CREATE MATERIALIZED VIEW secondvalue AS
  WITH stg1 AS (
      SELECT
        chart.*,
        lib.fio2           AS fio2_lib,
        lib.po2            AS po2_lib,
        lib.peep           AS peep_lib,
        gcs.gcsmotor,
        gcs.gcseyes,
        gcs.gcsverbal_plus AS gcsverbal
      FROM secondchart chart
        LEFT JOIN secondlab lib
          ON chart.icustay_id = lib.icustay_id
             AND chart.charttime = lib.charttime
        LEFT JOIN secondgcs gcs
          ON chart.icustay_id = gcs.icustay_id
             AND chart.charttime = gcs.charttime)
  SELECT
    subject_id,
    hadm_id,
    icustay_id,
    charttime,
    spo2,
    coalesce(po2_lib, pao2)  AS pao2,
    coalesce(fio2_lib, fio2) AS fio2,
    hr,
    temp,
    nbps,
    nbpd,
    nbpm,
    abps,
    abpd,
    abpm,
    rr,
    tv,
    pip,
    plap,
    mv,
    map,
    coalesce(peep_lib, peep) AS peep,
    gcsmotor,
    gcsverbal,
    gcseyes
  FROM stg1
