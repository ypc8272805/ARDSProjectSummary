/*
Author:ypc
Date:2018.01.29
Table:icustays labevents
Description:
本视图的目的是为了筛选患者，有一个条件是患者第一天的P/F值小于300，为了实现这个条件写的。

这个视图通过通过labevents表提取患者的实验室数据PaO2，FiO2,氧流量。
这里没有使用chartevents，是由于chartevents中的数据不一定可靠，对于实验指标建议使用labevents，但是对于FiO2，其实是使用了
labevents和chartevents的，但是这里看不出来，我是在下一步工作中将两个表的FiO2进行合并的。
 */
CREATE TABLE po2oneday AS WITH pvt AS (
    SELECT
      ie.subject_id,
      ie.hadm_id,
      ie.icustay_id,
      CASE
      WHEN (le.itemid = 50815)
        THEN 'O2FLOW' :: TEXT
      WHEN (le.itemid = 50816)
        THEN 'FIO2' :: TEXT
      WHEN (le.itemid = 50821)
        THEN 'PO2' :: TEXT
      ELSE NULL :: TEXT
      END AS label,
      le.charttime,
      le.value,
      CASE
      WHEN (le.valuenum <= (0) :: DOUBLE PRECISION)
        THEN NULL :: DOUBLE PRECISION
      WHEN ((le.itemid = 50815) AND (le.valuenum > (70) :: DOUBLE PRECISION))
        THEN NULL :: DOUBLE PRECISION
      WHEN ((le.itemid = 50821) AND (le.valuenum > (800) :: DOUBLE PRECISION))
        THEN NULL :: DOUBLE PRECISION
      WHEN ((le.itemid = 50816) AND (le.valuenum > (100) :: DOUBLE PRECISION) AND
            (le.valuenum < (21) :: DOUBLE PRECISION))
        THEN NULL :: DOUBLE PRECISION
      ELSE le.valuenum
      END AS valuenum
    FROM (mimiciii.icustays ie
      LEFT JOIN mimiciii.labevents le ON (((le.subject_id = ie.subject_id) AND (le.hadm_id = ie.hadm_id) AND
                                           ((le.charttime >= (ie.intime - '06:00:00' :: INTERVAL HOUR)) AND
                                            (le.charttime <= (ie.intime + '1 days' :: INTERVAL DAY))) AND
                                           (le.itemid = ANY (ARRAY [50815, 50816, 50821])))))
)
SELECT
  pvt.subject_id,
  pvt.hadm_id,
  pvt.icustay_id,
  pvt.charttime,
  max(
      CASE
      WHEN (pvt.label = 'O2FLOW' :: TEXT)
        THEN pvt.valuenum
      ELSE NULL :: DOUBLE PRECISION
      END) AS o2flow,
  max(
      CASE
      WHEN (pvt.label = 'FIO2' :: TEXT)
        THEN pvt.valuenum
      ELSE NULL :: DOUBLE PRECISION
      END) AS fio2,
  max(
      CASE
      WHEN (pvt.label = 'PO2' :: TEXT)
        THEN pvt.valuenum
      ELSE NULL :: DOUBLE PRECISION
      END) AS po2
FROM pvt
GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime
ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime;
