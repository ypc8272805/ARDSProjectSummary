-- ITEMIDs used:

-- CAREVUE
--    723 as GCSVerbal
--    454 as GCSMotor
--    184 as GCSEyes

-- METAVISION
--    223900 GCS - Verbal Response
--    223901 GCS - Motor Response
--    220739 GCS - Eye Opening

-- The code combines the ITEMIDs into the carevue itemids, then pivots those
-- So 223900 is changed to 723, then the ITEMID 723 is pivoted to form GCSVerbal

-- Note:
--  The GCS for sedated patients is defaulted to 15 in this code.
--  This is in line with how the data is meant to be collected.
--  e.g., from the SAPS II publication:
--    For sedated patients, the Glasgow Coma Score before sedation was used.
--    This was ascertained either from interviewing the physician who ordered the sedation,
--    or by reviewing the patient's medical record.

-- DROP MATERIALIZED VIEW IF EXISTS secondgcs CASCADE;
create materialized view secondgcs as
WITH base AS
(
    SELECT
      pvt.ICUSTAY_ID,
      pvt.charttime

      -- Easier names - note we coalesced Metavision and CareVue IDs below
      ,
      max(CASE WHEN pvt.itemid = 454
        THEN pvt.valuenum
          ELSE NULL END)             AS GCSMotor,
      max(CASE WHEN pvt.itemid = 723
        THEN pvt.valuenum
          ELSE NULL END)             AS GCSVerbal,
      max(CASE WHEN pvt.itemid = 184
        THEN pvt.valuenum
          ELSE NULL END)             AS GCSEyes

      -- If verbal was set to 0 in the below select, then this is an intubated patient
      ,
      CASE
      WHEN max(CASE WHEN pvt.itemid = 723
        THEN pvt.valuenum
               ELSE NULL END) = 0
        THEN 1
      ELSE 0
      END                            AS EndoTrachFlag,
      ROW_NUMBER()
      OVER (
        PARTITION BY pvt.ICUSTAY_ID
        ORDER BY pvt.charttime ASC ) AS rn

    FROM (
           SELECT
             l.ICUSTAY_ID
             -- merge the ITEMIDs so that the pivot applies to both metavision/carevue data
             ,
             CASE
             WHEN l.ITEMID IN (723, 223900)
               THEN 723
             WHEN l.ITEMID IN (454, 223901)
               THEN 454
             WHEN l.ITEMID IN (184, 220739)
               THEN 184
             ELSE l.ITEMID END
               AS ITEMID

             -- convert the data into a number, reserving a value of 0 for ET/Trach
             ,
             CASE
             -- endotrach/vent is assigned a value of 0, later parsed specially
             WHEN l.ITEMID = 723 AND l.VALUE = '1.0 ET/Trach'
               THEN 0 -- carevue
             WHEN l.ITEMID = 223900 AND l.VALUE = 'No Response-ETT'
               THEN 0 -- metavision

             ELSE VALUENUM
             END
               AS VALUENUM,
             l.CHARTTIME
           FROM CHARTEVENTS l

             -- get intime for charttime subselection
             INNER JOIN icustays b
               ON l.icustay_id = b.icustay_id

           -- Isolate the desired GCS variables
           WHERE l.ITEMID IN
                 (
                   -- 198 -- GCS
                   -- GCS components, CareVue
                   184, 454, 723
                   -- GCS components, Metavision
                   , 223900, 223901, 220739
                 )
                 -- Only get data for the first 24 hours
                 AND l.charttime BETWEEN b.intime AND b.outtime
                 -- exclude rows marked as error
                 AND l.error IS DISTINCT FROM 1
         ) pvt
    GROUP BY pvt.ICUSTAY_ID, pvt.charttime
),
    gcs AS (
      SELECT
        b.*,
        b2.GCSVerbal AS GCSVerbalPrev,
        b2.GCSMotor  AS GCSMotorPrev,
        b2.GCSEyes   AS GCSEyesPrev
      -- Calculate GCS, factoring in special case when they are intubated and prev vals
      -- note that the coalesce are used to implement the following if:
      --  if current value exists, use it
      --  if previous value exists, use it
      --  otherwise, default to normal
      --   , case
      -- --       -- replace GCS during sedation with 15
      -- --       when b.GCSVerbal = 0
      -- --         then 15
      --        when b.GCSVerbal is null and b2.GCSVerbal = 0
      --          then
      -- if previously they were intub, but they aren't now, do not use previous GCS values
      --       when b2.GCSVerbal = 0
      --         then
      --             coalesce(b.GCSMotor,6)
      --           + coalesce(b.GCSVerbal,5)
      --           + coalesce(b.GCSEyes,4)
      -- otherwise, add up score normally, imputing previous value if none available at current time
      --       else
      --             coalesce(b.GCSMotor,coalesce(b2.GCSMotor,6))
      --           + coalesce(b.GCSVerbal,coalesce(b2.GCSVerbal,5))
      --           + coalesce(b.GCSEyes,coalesce(b2.GCSEyes,4))
      --       end as GCS
      FROM base b
        -- join to itself within 6 hours to get previous value
        LEFT JOIN base b2
          ON b.ICUSTAY_ID = b2.ICUSTAY_ID AND b.rn = b2.rn + 1 AND b2.charttime > b.charttime - INTERVAL '6' HOUR
  ), gcs1 AS (
    SELECT
      icustay_id,
      charttime,
      CASE WHEN (GCSMotor = 0 OR GCSMotor ISNULL) AND GCSMotorPrev IS NOT NULL
        THEN GCSMotorPrev
      ELSE GCSMotor
      END AS gcsmotor,
      CASE WHEN (GCSVerbal = 0 OR GCSVerbal ISNULL) AND GCSVerbalPrev IS NOT NULL
        THEN GCSVerbalPrev
      ELSE GCSVerbal
      END AS gcsverbal,
      CASE WHEN (GCSEyes = 0 OR GCSEyes ISNULL) AND GCSEyesPrev IS NOT NULL
        THEN GCSEyesPrev
      ELSE GCSEyes
      END AS gcseyes

    FROM gcs)

SELECT
  *,
  CASE WHEN gcsverbal = 0 OR gcsverbal ISNULL
    THEN round(-0.3756 + gcsmotor * 0.5713 + gcseyes * 0.4233)
  ELSE gcsverbal
  END AS gcsverbal_plus
FROM gcs1



