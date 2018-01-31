/*
Author:ypc
Date:2018.01.29
Table:patients icustays ventfirstday noteevents chest_num heightweight icustay_detail
Description:
这个视图是筛选患者的一部分，通过筛选条件，一步一步进行选择患者，这个视图选择过程前期有一个notebook版本，用于展示如何开展
工作的，当然这种选择患者的思路也是借鉴了mimic-III团队在github上的教程，如果想了解具体思路我建议大家看一下github上的
notebook教程。
 */
--获取患者的icu基本信息，如患者在ICU停留的时间，计算患者的年龄(数据库中没有直接记录患者的年龄)，并标注是第几次 进入ICU
CREATE MATERIALIZED VIEW select_patient AS WITH icu_info AS (
    SELECT
      icu.subject_id,
      icu.hadm_id,
      icu.icustay_id,
      icu.intime,
      icu.outtime,
      vent.vent,
      (((date_part('epoch' :: TEXT, (icu.outtime - icu.intime)) / (60) :: DOUBLE PRECISION) / (60) :: DOUBLE PRECISION)
       / (24) :: DOUBLE PRECISION)                                 AS icu_length_of_stay,
      ((((date_part('epoch' :: TEXT, (icu.intime - pat.dob)) / (60) :: DOUBLE PRECISION) / (60) :: DOUBLE PRECISION) /
        (24) :: DOUBLE PRECISION) / (365.242) :: DOUBLE PRECISION) AS age,
      rank()
      OVER (
        PARTITION BY icu.subject_id
        ORDER BY icu.intime )                                      AS icustay_id_order
    FROM ((mimiciii.icustays icu
      JOIN mimiciii.patients pat ON ((icu.subject_id = pat.subject_id)))
      LEFT JOIN mimiciii.ventfirstday vent ON ((icu.icustay_id = vent.icustay_id)))
), chest_num AS (
  --提取患者的胸部影像学检查报告
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
)
--根据以上信息，来判断患者的对应条件
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
  LEFT JOIN mimiciii.icustay_detail icu_d ON ((icu_info.icustay_id = icu_d.icustay_id)));
