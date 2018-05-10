--需要提取患者的个人相关信息
--年龄 性别 身高 体重 种族 ICU类型 是否转移其他的ICU  cv和mv系统的区别

CREATE MATERIALIZED VIEW secondData AS
  WITH stg1 AS (
      SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.icustay_id,
        icu.first_careunit,
        icu.last_careunit,
        icu.dbsource,
        det.age,
        det.ethnicity,
        det.admission_type,
        det.gender,
        hw.height_first,
        hw.height_min,
        hw.height_max,
        hw.weight_first,
        hw.weight_min,
        hw.weight_max
      FROM icustays icu
        LEFT JOIN icustay_detail det
          ON icu.icustay_id = det.icustay_id
        LEFT JOIN heightweight hw
          ON icu.icustay_id = hw.icustay_id
  )
  SELECT
    secondmatch.*,
    stg1.first_careunit,
    stg1.last_careunit,
    stg1.dbsource,
    stg1.age,
    stg1.ethnicity,
    stg1.admission_type,
    stg1.gender,
    stg1.height_first,
    stg1.height_min,
    stg1.height_max,
    stg1.weight_first,
    stg1.weight_min,
    stg1.weight_max
  FROM secondmatch
    LEFT JOIN stg1
      ON secondmatch.icustay_id = stg1.icustay_id
