# """
# db.py — PostgreSQL connection for IPL NEXUS

# Credential priority:
#   1. st.secrets  → Streamlit Cloud (production)
#   2. os.getenv() → local environment variables

# FIX: Uses Transaction Pooler (port 6543) instead of Session Pooler
#      because psycopg2 prepared statements are NOT supported by Session Pooler.
#      Transaction Pooler supports all query types correctly.
# """

# import os
# import psycopg2
# import psycopg2.pool
# import pandas as pd
# import streamlit as st
# from typing import Optional, Tuple


# def _get_db_config() -> dict:
#     """
#     Load DB credentials from st.secrets (Streamlit Cloud)
#     or fall back to environment variables.

#     IMPORTANT: Uses Transaction Pooler (port 6543) for Supabase.
#     Session Pooler (port 5432) does NOT work with psycopg2.
#     """
#     # ── Priority 1: st.secrets (Streamlit Cloud) ──────────────────────────────
#     try:
#         secrets = st.secrets["postgres"]
#         return {
#             "host":     secrets["host"],
#             "port":     int(secrets.get("port", 6543)),
#             "dbname":   secrets["dbname"],
#             "user":     secrets["user"],
#             "password": secrets["password"],
#             "sslmode":  "require",
#             "options":  "-c statement_cache_size=0",
#         }
#     except (KeyError, FileNotFoundError):
#         pass

#     # ── Priority 2: Transaction Pooler fallback ────────────────────────────────
#     return {
#         "host":     os.getenv("PG_HOST",     "aws-1-ap-south-1.pooler.supabase.com"),
#         "port":     int(os.getenv("PG_PORT", "6543")),
#         "dbname":   os.getenv("PG_DB",       "postgres"),
#         "user":     os.getenv("PG_USER",     "postgres.qxgxodpethnqmwheyepq"),
#         "password": os.getenv("PG_PASSWORD", "Vaishnani@2728"),
#         "sslmode":  "require",
#         "options":  "-c statement_cache_size=0",
#     }


# # ── Connection — NO pool for Supabase Transaction Pooler ──────────────────────
# # ThreadedConnectionPool does NOT work well with Transaction Pooler.
# # Use direct connection per query instead.
# @st.cache_resource(show_spinner=False)
# def _get_config_cached():
#     """Cache the config so we don't re-read secrets every query."""
#     return _get_db_config()


# def run_query(sql: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
#     """
#     Execute a SQL SELECT query and return (DataFrame, error_string).
#     Uses a fresh connection per query — required for Transaction Pooler.
#     """
#     conn = None
#     try:
#         config = _get_config_cached()
#         conn = psycopg2.connect(**config)
#         conn.autocommit = True

#         with conn.cursor() as cur:
#             cur.execute(sql)
#             if cur.description is None:
#                 return pd.DataFrame(), None
#             cols = [desc[0] for desc in cur.description]
#             rows = cur.fetchmany(10_000)
#             df   = pd.DataFrame(rows, columns=cols)
#             for col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors="ignore")
#             return df, None

#     except psycopg2.OperationalError as e:
#         return None, f"Cannot connect to database: {str(e)}"
#     except psycopg2.Error as e:
#         return None, f"SQL Error: {e.pgerror or str(e)}"
#     except Exception as e:
#         return None, f"Unexpected error: {str(e)}"
#     finally:
#         if conn:
#             try:
#                 conn.close()
#             except Exception:
#                 pass


# def test_connection() -> bool:
#     """Quick ping — used by Test DB Connection button."""
#     df, err = run_query("SELECT 1 AS ok")
#     return err is None


# # ── Schema context sent to Groq LLM ───────────────────────────────────────────
# SCHEMA_CONTEXT = """
# PostgreSQL database: postgres (Supabase)
# EXACT table names (case-sensitive): "Matches" (capital M), "Players" (capital P), deliveries, innings, player_teams

# "Matches"(
#     match_id, season, match_date, match_city, match_venue,
#     toss_winner, toss_decision, match_type,
#     team1, team2, player_of_match, balls_per_over, overs,
#     winner, win_by_runs, win_by_wickets,
#     result, eliminator, match_key
# )

# innings(
#     innings_id, match_id, innings_number,
#     batting_team, bowling_team,
#     total_runs, total_wickets, total_balls, total_overs,
#     run_rate, target_runs, target_overs
# )

# deliveries(
#     delivery_id, match_id, inning_number,
#     over_number, ball_number,
#     batter_id, bowler_id, non_striker_id,
#     runs_batter, runs_extras, runs_total,
#     is_wicket, dismissal_type, player_out_id
# )

# "Players"(player_id, player_name, registry_id)

# player_teams(player_id, team_name, season)

# Key relationships:
# - deliveries.batter_id     → "Players".player_id
# - deliveries.bowler_id     → "Players".player_id
# - deliveries.match_id      → "Matches".match_id
# - innings.match_id         → "Matches".match_id

# CRITICAL DATA FACTS:
# - is_wicket stores 't' or 'f' (STRING) — use WHERE is_wicket = 't'
# - over_number range is 1 to 20 (NOT 0 to 19)
# - player_of_match format is {"Name"} — clean with REPLACE()
# """