/*
  Gera deploy/d.csv — Phase 2 retreino com filtro agentes + zeros.

  Saída: 10 colunas, sep=;, decimal=, sem header.
  banco_origem;dia;hora;dia_semana;total_dia;acumulado_ate_hora;
  proporcao_ate_hora;data_ultimo_batimento;dias_desde_ultimo_batimento;faixa_batimento

  Regras:
  - Bancos: AUTOS + CONSUMER (DB schemas separados)
  - Janela: 2025-11-07 → 2026-05-08 (dias úteis)
  - Horas 8-19, dia_semana 2=seg…6=sex
  - Acordo válido: ID_REC_STATUS IN (1,3,12) AND PARCELA=0
  - Filtro agentes completo (8 condições, UPPER+TRIM)
  - Inclui zeros (CROSS JOIN calendar × horas × banco)
  - faixa_batimento: 0-5 pos_batimento; 6-15 absorcao; >15/NULL basal

  Export SSMS: Results → Save Results As → CSV
    Tools → Options → Query Results → SQL Server → Results to Grid:
      Include column headers when copying = OFF
      Default separator = ; (configurar antes)
    Ou: SQLCMD -S server -d db -Q "..." -o d.csv -s ";" -W -h -1
*/

SET DATEFIRST 7;  -- DOM=1, SEG=2, ... SAB=7

DECLARE @data_ini DATE = '2025-11-07';
DECLARE @data_fim DATE = '2026-05-08';

WITH
-- Calendário dias úteis na janela
Calendario AS (
    SELECT CAST(@data_ini AS DATE) AS dia
    UNION ALL
    SELECT DATEADD(DAY, 1, dia)
    FROM Calendario
    WHERE dia < @data_fim
),
DiasUteis AS (
    SELECT dia, DATEPART(WEEKDAY, dia) AS dia_semana
    FROM Calendario
    WHERE DATEPART(WEEKDAY, dia) BETWEEN 2 AND 6  -- seg–sex
),
Horas AS (
    SELECT 8 AS hora UNION ALL SELECT 9  UNION ALL SELECT 10 UNION ALL
    SELECT 11      UNION ALL SELECT 12 UNION ALL SELECT 13 UNION ALL
    SELECT 14      UNION ALL SELECT 15 UNION ALL SELECT 16 UNION ALL
    SELECT 17      UNION ALL SELECT 18 UNION ALL SELECT 19
),
Bancos AS (
    SELECT 'COBwebRCBAUTOS' AS banco_origem UNION ALL SELECT 'COBwebRCBCONSUMER'
),
Grade AS (  -- toda combinação banco × dia útil × hora
    SELECT b.banco_origem, d.dia, d.dia_semana, h.hora
    FROM Bancos b
    CROSS JOIN DiasUteis d
    CROSS JOIN Horas h
),

-- Acordos CONSUMER com filtro agentes completo
AcordosCONSUMER AS (
    SELECT
        CAST(R.DT_EMISSAO AS DATE) AS dia,
        DATEPART(HOUR, R.DT_EMISSAO) AS hora,
        COUNT(DISTINCT R.NR_RECEBIMENTO) AS qtd
    FROM COBwebRCBCONSUMER..REC_MASTER R WITH (NOLOCK)
    INNER JOIN COBwebRCBCONSUMER..USU_MASTER U WITH (NOLOCK)
        ON U.ID_USUARIO = R.ID_USUARIO
    WHERE R.DT_EMISSAO >= @data_ini
      AND R.DT_EMISSAO <  DATEADD(DAY, 1, @data_fim)
      AND R.ID_REC_STATUS IN (1, 3, 12)
      AND R.PARCELA = 0
      AND DATEPART(HOUR, R.DT_EMISSAO) BETWEEN 8 AND 19
      AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'COBDESANTOS'
      AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'NEMBUSUSER'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) <> 'NEMBUSUSER'
      AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'ANTLIA%'
      AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'INTERNA%'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'INTERNA%'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SUPORTE%'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SISTEMA%'
    GROUP BY CAST(R.DT_EMISSAO AS DATE), DATEPART(HOUR, R.DT_EMISSAO)
),
-- Acordos AUTOS — mesmo filtro
AcordosAUTOS AS (
    SELECT
        CAST(R.DT_EMISSAO AS DATE) AS dia,
        DATEPART(HOUR, R.DT_EMISSAO) AS hora,
        COUNT(DISTINCT R.NR_RECEBIMENTO) AS qtd
    FROM COBwebRCBAUTOS..REC_MASTER R WITH (NOLOCK)
    INNER JOIN COBwebRCBAUTOS..USU_MASTER U WITH (NOLOCK)
        ON U.ID_USUARIO = R.ID_USUARIO
    WHERE R.DT_EMISSAO >= @data_ini
      AND R.DT_EMISSAO <  DATEADD(DAY, 1, @data_fim)
      AND R.ID_REC_STATUS IN (1, 3, 12)
      AND R.PARCELA = 0
      AND DATEPART(HOUR, R.DT_EMISSAO) BETWEEN 8 AND 19
      AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'COBDESANTOS'
      AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'NEMBUSUSER'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) <> 'NEMBUSUSER'
      AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'ANTLIA%'
      AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'INTERNA%'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'INTERNA%'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SUPORTE%'
      AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SISTEMA%'
    GROUP BY CAST(R.DT_EMISSAO AS DATE), DATEPART(HOUR, R.DT_EMISSAO)
),
-- União com banco_origem + LEFT JOIN na grade pra incluir zeros
AcordosCompleto AS (
    SELECT g.banco_origem, g.dia, g.dia_semana, g.hora,
           COALESCE(a.qtd, 0) AS qtd
    FROM Grade g
    LEFT JOIN (
        SELECT 'COBwebRCBCONSUMER' AS banco_origem, dia, hora, qtd FROM AcordosCONSUMER
        UNION ALL
        SELECT 'COBwebRCBAUTOS'    AS banco_origem, dia, hora, qtd FROM AcordosAUTOS
    ) a
      ON a.banco_origem = g.banco_origem
     AND a.dia          = g.dia
     AND a.hora         = g.hora
),
-- total_dia + acumulado_ate_hora por (banco, dia)
ComAgregados AS (
    SELECT
        banco_origem, dia, dia_semana, hora, qtd,
        SUM(qtd) OVER (PARTITION BY banco_origem, dia) AS total_dia,
        SUM(qtd) OVER (PARTITION BY banco_origem, dia ORDER BY hora
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS acumulado_ate_hora
    FROM AcordosCompleto
),
-- Batimentos por banco
BatimentosCONSUMER AS (
    SELECT DISTINCT CAST([DATA] AS DATE) AS dia_batimento
    FROM COBwebRCBCONSUMER..CARGA_LOTE WITH (NOLOCK)
    WHERE ID_USUARIO = 1 AND QTD_NV_CLI > 10000
),
BatimentosAUTOS AS (
    SELECT DISTINCT CAST([DATA] AS DATE) AS dia_batimento
    FROM COBwebRCBAUTOS..CARGA_LOTE WITH (NOLOCK)
    WHERE ID_USUARIO = 1 AND QTD_NV_CLI > 10000
)

SELECT
    a.banco_origem,
    CONVERT(VARCHAR(10), a.dia, 23)        AS dia,
    a.hora,
    a.dia_semana,
    a.total_dia,
    a.acumulado_ate_hora,
    CASE WHEN a.total_dia = 0 THEN 0.0
         ELSE CAST(a.acumulado_ate_hora AS FLOAT) / a.total_dia
    END                                     AS proporcao_ate_hora,
    CONVERT(VARCHAR(10), b.dia_batimento, 23) AS data_ultimo_batimento,
    DATEDIFF(DAY, b.dia_batimento, a.dia)     AS dias_desde_ultimo_batimento,
    CASE
        WHEN b.dia_batimento IS NULL THEN 'basal'
        WHEN DATEDIFF(DAY, b.dia_batimento, a.dia) BETWEEN 0 AND 5  THEN 'pos_batimento'
        WHEN DATEDIFF(DAY, b.dia_batimento, a.dia) BETWEEN 6 AND 15 THEN 'absorcao'
        ELSE 'basal'
    END                                     AS faixa_batimento
FROM ComAgregados a
OUTER APPLY (
    SELECT TOP 1 bb.dia_batimento
    FROM (
        SELECT dia_batimento FROM BatimentosCONSUMER
         WHERE a.banco_origem = 'COBwebRCBCONSUMER'
        UNION ALL
        SELECT dia_batimento FROM BatimentosAUTOS
         WHERE a.banco_origem = 'COBwebRCBAUTOS'
    ) bb
    WHERE bb.dia_batimento <= a.dia
    ORDER BY bb.dia_batimento DESC
) b
ORDER BY a.banco_origem, a.dia, a.hora
OPTION (MAXRECURSION 400);
