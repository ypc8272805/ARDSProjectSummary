DROP MATERIALIZED VIEW secondmatch
CREATE MATERIALIZED VIEW secondmatch AS
--首先是将所有参数从patientvalue中提取出来，每一个参数创建一个临时表，方便后面使用，但是这样写很繁琐，应该有更好的方法，
-- 但是目前我还没有想到。
WITH pao2 AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      pao2
    FROM secondvalue
    WHERE pao2 IS NOT NULL
), spo2 AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      spo2
    FROM secondvalue
    WHERE secondvalue.spo2 IS NOT NULL
), fio2 AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      fio2
    FROM secondvalue
    WHERE fio2 IS NOT NULL
), hr AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      hr
    FROM secondvalue
    WHERE hr IS NOT NULL
), temp AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      temp
    FROM secondvalue
    WHERE secondvalue.temp IS NOT NULL
), nbps AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      nbps
    FROM secondvalue
    WHERE secondvalue.nbps IS NOT NULL
), nbpd AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      nbpd
    FROM secondvalue
    WHERE secondvalue.nbpd IS NOT NULL
), nbpm AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      nbpm
    FROM secondvalue
    WHERE secondvalue.nbpm IS NOT NULL
), abps AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      abps
    FROM secondvalue
    WHERE secondvalue.abps IS NOT NULL
), abpd AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      abpd
    FROM secondvalue
    WHERE secondvalue.abpd IS NOT NULL
), abpm AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      abpm
    FROM secondvalue
    WHERE secondvalue.abpm IS NOT NULL
), rr AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      rr
    FROM secondvalue
    WHERE secondvalue.rr IS NOT NULL
), tv AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      tv
    FROM secondvalue
    WHERE secondvalue.tv IS NOT NULL
), pip AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      pip
    FROM secondvalue
    WHERE secondvalue.pip IS NOT NULL
), plap AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      plap
    FROM secondvalue
    WHERE secondvalue.plap IS NOT NULL
), mv AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      mv
    FROM secondvalue
    WHERE secondvalue.mv IS NOT NULL
), map AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      map
    FROM secondvalue
    WHERE secondvalue.map IS NOT NULL
), peep AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      peep
    FROM secondvalue
    WHERE secondvalue.peep IS NOT NULL
), GCSm AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      gcsmotor
    FROM secondvalue
    WHERE secondvalue.gcsmotor IS NOT NULL
), GCSv AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      gcsverbal
    FROM secondvalue
    WHERE secondvalue.gcsverbal IS NOT NULL
), GCSe AS (
    SELECT
      subject_id,
      hadm_id,
      icustay_id,
      charttime,
      gcseyes
    FROM secondvalue
    WHERE secondvalue.gcseyes IS NOT NULL
), patvalue AS (
  --以pao2为标准，匹配2小时以前的spo2数据，如果没有就为空，如果有多条，选择最近的一条，也就是选择lastRowSpO2=1的就OK了
    SELECT
      pao2.subject_id,
      pao2.hadm_id,
      pao2.icustay_id,
      pao2.charttime,
      pao2.pao2,
      spo2.spo2,
      spo2.charttime                           AS s_time,
      CASE WHEN pao2.charttime > spo2.charttime
        THEN pao2.charttime - spo2.charttime
      ELSE spo2.charttime - pao2.charttime END AS sc
    FROM pao2
      LEFT JOIN spo2
        ON pao2.icustay_id = spo2.icustay_id
           AND ((spo2.charttime BETWEEN pao2.charttime - INTERVAL '4' HOUR AND pao2.charttime)
                OR (spo2.charttime BETWEEN pao2.charttime AND pao2.charttime + INTERVAL '4' HOUR))
), patvalue_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowSpO2
    FROM patvalue
), stg1 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY fio2.charttime DESC ) AS lastRowFiO2,
      fio2.fio2
    FROM patvalue_1 pat
      LEFT JOIN fio2
        ON fio2.icustay_id = pat.icustay_id
           AND fio2.charttime BETWEEN pat.charttime - INTERVAL '12' HOUR AND pat.charttime
    WHERE pat.lastRowSpO2 = 1
), stg2 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowHR
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.hr
    FROM stg1 pat
      LEFT JOIN hr s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowFiO2 = 1
), stg2_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowHR
    FROM stg2
), stg3 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowTemp
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.temp
    FROM stg2_1 pat
      LEFT JOIN temp s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowHR = 1
), stg3_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowTemp
    FROM stg3
), stg4 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowNbps

      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC ) AS lastRownbps,
      s2.nbps
    FROM stg3_1 pat
      LEFT JOIN nbps s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime

    WHERE pat.lastRowTemp = 1
), stg5 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC ) AS lastRowNbpd,
      s2.nbpd
    FROM stg4 pat
      LEFT JOIN nbpd s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime

    WHERE pat.lastRowNbps = 1
), stg6 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC ) AS lastRowNbpm,
      s2.nbpm
    FROM stg5 pat
      LEFT JOIN nbpm s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime

    WHERE pat.lastRowNbpd = 1
), stg7 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC ) AS lastRowAbps,
      s2.abps
    FROM stg6 pat
      LEFT JOIN abps s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime

    WHERE pat.lastRowNbpm = 1
), stg8 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC )        AS lastRowAbpd,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.abpd
    FROM stg7 pat
      LEFT JOIN abpd s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime

    WHERE pat.lastRowAbps = 1
), stg9 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC ) AS lastRowAbpm,
      s2.abpm
    FROM stg8 pat
      LEFT JOIN abpm s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime

    WHERE pat.lastRowAbpd = 1
), stg10 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowRR
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.rr
    FROM stg9 pat
      LEFT JOIN rr s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowAbpm = 1
), stg10_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowRR
    FROM stg10
), stg11 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowTV
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.tv
    FROM stg10_1 pat
      LEFT JOIN tv s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowRR = 1
), stg11_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowTV
    FROM stg11
), stg12 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr,
      pat.tv
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowPIP
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.pip
    FROM stg11_1 pat
      LEFT JOIN pip s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowTV = 1
), stg12_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowPIP
    FROM stg12
), stg13 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr,
      pat.tv,
      pat.pip
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowPLAP
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.plap
    FROM stg12_1 pat
      LEFT JOIN plap s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowPIP = 1
), stg13_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowPLAP
    FROM stg13
), stg14 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr,
      pat.tv,
      pat.pip,
      pat.plap
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowMV
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.mv
    FROM stg13_1 pat
      LEFT JOIN mv s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowPLAP = 1
), stg14_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowMV
    FROM stg14
), stg15 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr,
      pat.tv,
      pat.pip,
      pat.plap,
      pat.mv
      --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowMAP
      ,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.map
    FROM stg14_1 pat
      LEFT JOIN map s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowMV = 1
), stg15_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowMAP
    FROM stg15
), stg16 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr,
      pat.tv,
      pat.pip,
      pat.plap,
      pat.mv,
      pat.map,
      row_number()
      OVER (
        PARTITION BY pat.icustay_id, pat.charttime
        ORDER BY s2.charttime DESC ) AS lastRowPEEP,
      s2.peep
    FROM stg15_1 pat
      LEFT JOIN peep s2
        ON s2.icustay_id = pat.icustay_id
           AND s2.charttime BETWEEN pat.charttime - INTERVAL '12' HOUR AND pat.charttime
    WHERE pat.lastRowMAP = 1
),
  --       stg16_1 AS (
  --       SELECT
  --         *,
  --         row_number()
  --         OVER (
  --           PARTITION BY icustay_id, charttime
  --           ORDER BY sc ) AS lastRowPEEP
  --       FROM stg16
  --   ),
    stg17 AS
  (SELECT
     pat.subject_id,
     pat.hadm_id,
     pat.icustay_id,
     pat.charttime,
     pat.pao2,
     pat.spo2,
     pat.fio2,
     pat.hr,
     pat.temp,
     pat.nbps,
     pat.nbpd,
     pat.nbpm,
     pat.abps,
     pat.abpd,
     pat.abpm,
     pat.rr,
     pat.tv,
     pat.pip,
     pat.plap,
     pat.mv,
     pat.map,
     pat.peep,
     CASE WHEN pat.charttime > s2.charttime
       THEN pat.charttime - s2.charttime
     ELSE s2.charttime - pat.charttime END AS sc,
     s2.gcsmotor
   FROM stg16 pat
     LEFT JOIN GCSm s2
       ON s2.icustay_id = pat.icustay_id
          AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
               OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
   WHERE pat.lastRowPEEP = 1
  ),
    stg17_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowGCSm
      FROM stg17
  ),
    stg18 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr,
        pat.tv,
        pat.pip,
        pat.plap,
        pat.mv,
        pat.map,
        pat.peep,
        pat.gcsmotor,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.gcsverbal
      FROM stg17_1 pat
        LEFT JOIN GCSv s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowGCSm = 1
  ), stg18_1 AS (
    SELECT
      *,
      row_number()
      OVER (
        PARTITION BY icustay_id, charttime
        ORDER BY sc ) AS lastRowGCSv
    FROM stg18
), stg19 AS (
    SELECT
      pat.subject_id,
      pat.hadm_id,
      pat.icustay_id,
      pat.charttime,
      pat.pao2,
      pat.spo2,
      pat.fio2,
      pat.hr,
      pat.temp,
      pat.nbps,
      pat.nbpd,
      pat.nbpm,
      pat.abps,
      pat.abpd,
      pat.abpm,
      pat.rr,
      pat.tv,
      pat.pip,
      pat.plap,
      pat.mv,
      pat.map,
      pat.peep,
      pat.gcsmotor,
      pat.gcsverbal,
      CASE WHEN pat.charttime > s2.charttime
        THEN pat.charttime - s2.charttime
      ELSE s2.charttime - pat.charttime END AS sc,
      s2.gcseyes
    FROM stg18_1 pat
      LEFT JOIN GCSe s2
        ON s2.icustay_id = pat.icustay_id
           AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
    WHERE pat.lastRowGCSv = 1
),
    stg19_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowGCSe
      FROM stg19
  )
SELECT
  pat.subject_id,
  pat.hadm_id,
  pat.icustay_id,
  pat.charttime,
  pat.pao2,
  pat.spo2,
  pat.fio2,
  pat.hr,
  pat.temp,
  pat.nbps,
  pat.nbpd,
  pat.nbpm,
  pat.abps,
  pat.abpd,
  pat.abpm,
  pat.rr,
  pat.tv,
  pat.pip,
  pat.plap,
  pat.mv,
  pat.map,
  pat.peep,
  pat.gcsmotor,
  pat.gcsverbal,
  pat.gcseyes
FROM stg19_1 pat
WHERE pat.lastRowGCSe = 1