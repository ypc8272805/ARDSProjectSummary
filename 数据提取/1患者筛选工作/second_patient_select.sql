CREATE MATERIALIZED VIEW secondPatientSelect AS
  WITH icu_info AS (
      SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.icustay_id,
        icu.intime,
        icu.outtime,
        vent.vent,
        extract(EPOCH FROM icu.outtime - icu.intime) / 60 / 60 / 24       AS icu_length_of_stay,
        extract(EPOCH FROM icu.intime - pat.dob) / 60 / 60 / 24 / 365.242 AS age,
        rank()
        OVER (
          PARTITION BY icu.subject_id
          ORDER BY icu.intime )                                           AS icustay_id_order
      FROM mimiciii.icustays icu
        JOIN mimiciii.patients pat
          ON icu.subject_id = pat.subject_id
        LEFT JOIN mimiciii.ventfirstday vent
          ON icu.icustay_id = vent.icustay_id

  ), chest_num AS (
      SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.icustay_id,
        count(*) AS chest_num
      FROM (mimiciii.icustays icu
        LEFT JOIN mimiciii.noteevents note ON ((icu.hadm_id = note.hadm_id)))
      WHERE ((note.charttime > icu.intime) AND (note.charttime < icu.outtime) AND
             ((note.category) :: TEXT = 'Radiology' :: TEXT) AND ((note.description) :: TEXT ~~ '%CHEST%' :: TEXT))
      GROUP BY icu.subject_id, icu.hadm_id, icu.icustay_id
  ), stg1 AS (
      SELECT
        icu_info.subject_id,
        icu_info.hadm_id,
        icu_info.icustay_id,
        icu_info.age,
        icu_d.gender,
        hw.height_first,
        hw.weight_first,
        icu_d.ethnicity,
        icu_info.intime,
        icu_info.outtime,
        icu_info.icu_length_of_stay,
        CASE
        WHEN (icu_info.icu_length_of_stay < (2) :: DOUBLE PRECISION)
          THEN 1
        ELSE 0
        END AS exclusion_los,
        CASE
        WHEN (icu_info.age < (16) :: DOUBLE PRECISION)
          THEN 1
        ELSE 0
        END AS exclusion_age,
        CASE
        WHEN (icu_info.icustay_id_order <> 1)
          THEN 1
        ELSE 0
        END AS exclusion_first_stay,
        CASE
        WHEN (icu_info.vent = 0)
          THEN 1
        ELSE 0
        END AS exclusion_vent,
        CASE
        WHEN (chest_num.chest_num < 1)
          THEN 1
        ELSE 0
        END AS exclusion_chest
      FROM (((icu_info
        LEFT JOIN chest_num ON ((chest_num.icustay_id = icu_info.icustay_id)))
        LEFT JOIN mimiciii.heightweight hw ON ((hw.icustay_id = icu_info.icustay_id)))
        LEFT JOIN mimiciii.icustay_detail icu_d ON ((icu_info.icustay_id = icu_d.icustay_id)))
  )
  SELECT
    stg1.*,
    pf_1222.exclusion_pf
  FROM stg1
    LEFT JOIN pf_1222
      ON stg1.icustay_id = pf_1222.icustay_id

