/*
Author:ypc
Date:2018.01.29
Table:
Description:
当患者的生理参数提取完之后，要筛选我们需要的，这里有一些要注意，要对一些异常值进行处理，这样在后续分析时更简单。
特别注意 ，这里之所以叫value_pf 是按照患者的epf来筛选患者，
还有一个文件叫value_icd是根据患者的icd来筛选患者，并提取参数的，
这两个 pf 筛选的观测值要大于 icd

 */
WITH allpatients AS (
  --选择需要的患者，其实这一部分应该是重复了，因为在提取参数的时候也做过限制了
    SELECT
      select_patient.subject_id,
      select_patient.hadm_id,
      select_patient.icustay_id,
      select_patient.intime,
      select_patient.outtime,
      select_patient.exclusion_age,
      select_patient.exclusion_chest,
      select_patient.exclusion_first_stay,
      select_patient.exclusion_los,
      select_patient.exclusion_vent,
      pf.exclusion_pf
    FROM select_patient_1222 select_patient
      LEFT JOIN pf_1222 AS pf
        ON select_patient.icustay_id = pf.icustay_id
), my_patients AS (
    SELECT *
    FROM allpatients
    WHERE
      exclusion_pf = 0 AND exclusion_vent = 0 AND exclusion_los = 0 AND exclusion_first_stay = 0 AND exclusion_chest = 0
      AND exclusion_age = 0
), stg1 AS (
    SELECT match.*
    FROM matchvalue_1222 match
    WHERE match.subject_id IN (
      SELECT match.subject_id
      FROM my_patients
    )
          --AND match.map is not null
          AND match.pao2 IS NOT NULL
          AND match.spo2 IS NOT NULL
          AND match.fio2 IS NOT NULL
          AND match.mv < 50

)
  , stg2 AS (
  --添加性别，计算一些衍生变量
    SELECT
      stg1.*,
      sp.weight_first,
      sp.height_first,
      sp.gender,
      sp.ethnicity,
      sp.age,
      CASE WHEN gender = 'F'
        THEN 1
      ELSE 0
      END                                     AS gender_num,
      CASE WHEN ethnicity LIKE '%WHITE%'
        THEN 1
      WHEN ethnicity LIKE '%BLACK%'
        THEN 2
      WHEN ethnicity LIKE '%HISPANIC%'
        THEN 3
      WHEN ethnicity LIKE '%ASIAN%'
        THEN 4
      ELSE 5
      END                                     AS eth_num,
      spo2 / (fio2 / 100)                     AS sf,
      pao2 / (fio2 / 100)                     AS pf,
      (map * fio2) / pao2                     AS OI,
      (map * fio2) / spo2                     AS OSI,
      weight_first / (height_first / 100) ^ 2 AS BMI
    FROM stg1
      LEFT JOIN select_patient_1222 sp
        ON stg1.icustay_id = sp.icustay_id
)
--为数据添加标签
SELECT
  *,
  CASE WHEN stg2.age > 300
    THEN 91.4
  ELSE stg2.age
  END AS age_new,
  CASE WHEN pf > 300
    THEN 1
  ELSE 0
  END AS two_class_300,
  CASE WHEN pf > 200
    THEN 1
  ELSE 0
  END AS two_class_200,
  CASE WHEN pf > 100
    THEN 1
  ELSE 0
  END AS two_class_100,
  CASE WHEN pf > 300
    THEN 1
  WHEN pf > 200
    THEN 2
  WHEN pf > 100
    THEN 3
  ELSE 4
  END AS four_class,
  CASE WHEN pf > 300
    THEN 1
  WHEN pf > 200
    THEN 2
  ELSE 3
  END AS three_class
FROM stg2
--去除一些异常值
WHERE stg2.height_first < 250
      AND stg2.height_first > 50
      AND stg2.tv < 2000