-- Run this entire file in the Supabase SQL Editor (one-time setup).
-- It creates the three tables the app needs and locks each one down
-- so users can only see and edit their own data.

-- ── Teams ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS teams (
    id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id    UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    name       TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE teams ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage their own teams"
    ON teams FOR ALL
    USING (auth.uid() = user_id);


-- ── Team configuration ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS team_config (
    id                       UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    team_id                  UUID REFERENCES teams(id) ON DELETE CASCADE UNIQUE NOT NULL,
    unit_of_work             TEXT    DEFAULT 'Point'  CHECK (unit_of_work IN ('Point', 'Issue')),
    analysis_mode            TEXT    DEFAULT 'Rolling' CHECK (analysis_mode IN ('Rolling', 'All')),
    sprints_per_window       INTEGER DEFAULT 5,
    strong_threshold         FLOAT   DEFAULT 0.5,
    moderate_threshold       FLOAT   DEFAULT 0.33,
    needs_attention_threshold FLOAT  DEFAULT 0.25,
    conservative_percentile  FLOAT   DEFAULT 0.15,
    trend_lookback           INTEGER DEFAULT 5,
    min_sprints_warning      INTEGER DEFAULT 10
);

ALTER TABLE team_config ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage config for their own teams"
    ON team_config FOR ALL
    USING (team_id IN (SELECT id FROM teams WHERE user_id = auth.uid()));


-- ── Sprint data ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sprint_data (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    team_id           UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    sprint_name       TEXT NOT NULL,
    sprint_date       DATE,
    completed_points  INTEGER DEFAULT 0,
    completed_issues  INTEGER DEFAULT 0,
    exclude           BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE sprint_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage sprint data for their own teams"
    ON sprint_data FOR ALL
    USING (team_id IN (SELECT id FROM teams WHERE user_id = auth.uid()));
