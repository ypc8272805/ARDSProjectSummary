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
WITH stg1 AS (
    SELECT match.*
    FROM matchvalue_1225 match
    WHERE match.subject_id IN (
      SELECT select_patient_1222.subject_id
      FROM select_patient_1222
      WHERE exclusion_vent = 0
            AND exclusion_los = 0
            AND exclusion_first_stay = 0
            AND exclusion_chest = 0
            AND exclusion_age = 0
            AND exclusion_icd = 0
      GROUP BY select_patient_1222.subject_id
    )
          --AND match.map is not null
          AND match.pao2 IS NOT NULL
          AND match.spo2 IS NOT NULL
          AND match.fio2 IS NOT NULL
          AND match.mv < 50

)
  , stg2 AS (
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
WHERE stg2.height_first < 250
      AND stg2.height_first > 50
      AND stg2.tv < 2000