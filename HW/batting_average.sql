use test;

-- create index on game and team_pitching_count table to make join faster
create or replace INDEX game_local_date_index
ON game (game_id,local_date)
;

create or replace INDEX game_id_pitching_index
ON team_pitching_counts (game_id)
;
-- joining game and team_pithcing_counts table to get local date
-- creating a column that defines if the team was the away team or home team
create or replace table joined_game_team_pitching_counts
 select  		g.local_date
                ,g.game_id
                ,tpc.team_id
                ,tpc.Single
                ,tpc.Double
                ,tpc.Triple
                ,tpc.atBat
                ,tpc.Home_Run
                ,tpc.Hit
                ,tpc.Walk
                ,tpc.Ground_Out
                ,tpc.Flyout
                ,tpc.Strikeout
                ,CASE when tpc.team_id = g.away_team_id then 'A'
                	when tpc.team_id = g.home_team_id then 'H'
                	else null end as home_or_away
 from team_pitching_counts tpc
 join game g on tpc.game_id=g.game_id;

-- creating index to make self join faster
create or replace INDEX game_id_local_date_pitching_index
ON joined_game_team_pitching_counts (game_id,local_date)
;
-- create rolling average of features extracted from team_pitching_counts table
create or replace table rolling_game_team_pitching
select t1.local_date as t1_date
        , t1.game_id
        , t1.team_id
        , t1.home_or_away
        , sum(t2.Single) as pitching_single
        , sum(t2.Double) as pitching_double
        , sum(t2.Triple) as pitching_triple
        , sum(t2.atBat) as pitching_atbat
        , sum(t2.Home_Run) as pitching_homerun
        , sum(t2.Hit) as pitching_hit
        , sum(t2.Walk) as pitching_baseonball_or_walk
        , IFNULL(sum(t2.ground_out)/sum(NULLIF(t2.Flyout,0)),0) as pitching_go_to_ao
        , sum(t2.ground_out) as pitching_groundout
        , sum(t2.Flyout) as pitching_flyout_or_airout
        , IFNULL(sum(t2.Strikeout)/sum(NULLIF(t2.Home_Run,0)),0) as pitching_so_to_hr
        , IFNULL(sum(t2.atBat)/sum(NULLIF(t2.Home_Run,0)),0) as pitching_ab_to_hr
        , IFNULL(sum(t2.atBat)/sum(NULLIF(t2.Hit,0)),0) as pitching_batting_average
    from joined_game_team_pitching_counts t1
    join joined_game_team_pitching_counts t2 ON
        t1.local_date > DATE_ADD(t2.local_date , INTERVAL -100 DAY) AND t1.team_id =t2.team_id
     group by t1.local_date, t1.team_id, t1.home_or_away
     order by t1.team_id, t1.local_date ASC , t2.local_date ASC;

-- create index on team_batting_counts to make the join faster
create or replace INDEX game_id_batting_index
ON team_batting_counts (game_id)
;
-- joining team_batting_counts with game table to get the local_date
create or replace table joined_game_team_batting_counts
 select  		  g.local_date
                , g.game_id
                , tbc.team_id
                , tbc.Single
                , tbc.Double
                , tbc.Triple
                , tbc.atBat
                , tbc.Home_Run
                , tbc.Hit
                , tbc.Walk
                , tbc.Hit_by_Pitch
                , tbc.Ground_Out
                , tbc.Flyout
                , tbc.Strikeout
                ,CASE when tbc.team_id = g.away_team_id then 'A'
                	when tbc.team_id = g.home_team_id then 'H'
                	else null end as home_or_away
    from team_batting_counts tbc
    join game g on tbc.game_id=g.game_id;

-- creating index to make self join faster
create or replace INDEX game_id_local_date_batting_index
ON joined_game_team_batting_counts (game_id,local_date)
;
-- create rolling average of features extracted from team_batting_counts table
create or replace table rolling_game_team_batting
select t1.local_date as t1_date
        , t1.game_id
        , t1.team_id
        , t1.home_or_away
        , sum(t2.Single) as batting_single
        , sum(t2.Double) as batting_double
        , sum(t2.Triple) as batting_triple
        , sum(t2.atBat) as batting_atbat
        , sum(t2.Home_Run) as batting_homerun
        , sum(t2.Hit) as batting_hit
        , SUM(t2.Hit_by_Pitch) as batting_hit_by_pitch
        , sum(t2.Walk) as batting_baseonball_or_walk
        , IFNULL(sum(t2.atBat)/sum(NULLIF(t2.Home_Run,0)),0) as batting_ab_to_hr
        , sum(t2.ground_out) as batting_groundout
        , sum(t2.Flyout) as batting_flyout_or_airout
        , IFNULL(sum(t2.Walk)/sum(NULLIF(t2.Strikeout,0)),0) as batting_w_to_sr
        , IFNULL(sum(t2.Ground_Out)/sum(NULLIF(t2.Flyout,0)),0) as batting_go_to_fo_or_ao
        , IFNULL(sum(t2.atBat)/sum(NULLIF(t2.Hit,0)),0) as batting_average_batting
        , IFNULL(sum(t2.Home_Run)/sum(NULLIF(t2.Hit,0)),0) as batting_hr_to_hit
        , sum(t2.Hit) + sum(t2.Walk) + sum(t2.Hit_by_Pitch) as times_on_base_or_tob
        , sum(t2.Single) + sum(t2.Double) + sum(t2.Triple) + sum(t2.Home_Run) as extra_base_hits_or_ebh
    from joined_game_team_batting_counts t1
    join joined_game_team_batting_counts t2 ON
        t1.local_date > DATE_ADD(t2.local_date , INTERVAL -100 DAY) AND t1.team_id =t2.team_id
     group by t1.local_date, t1.team_id, t1.home_or_away
     order by t1.team_id, t1.local_date ASC , t2.local_date ASC;

-- now join two tables to get one table, but create index on both to make it faster
create or replace INDEX team_game_batting_indx
ON rolling_game_team_batting (game_id,team_id)
;
create or replace INDEX team_game_pitching_indx
ON rolling_game_team_pitching (game_id,team_id)
;
-- join rolling_game_team_pitching and rolling_game_team_batting
CREATE or REPLACE TABLE joined_team_batting_pitching
 select rgtp.*
 		,rgtb.batting_single
 		,rgtb.batting_double
 		,rgtb.batting_triple
 		,rgtb.batting_atbat
 		,rgtb.batting_homerun
 		,rgtb.batting_hit
 		,rgtb.batting_hit_by_pitch
 		,rgtb.batting_baseonball_or_walk
 		,rgtb.batting_ab_to_hr
 		,rgtb.batting_groundout
 		,rgtb.batting_flyout_or_airout
 		,rgtb.batting_w_to_sr
 		,rgtb.batting_go_to_fo_or_ao
 		,rgtb.batting_average_batting
 		,rgtb.batting_hr_to_hit
 		,rgtb.times_on_base_or_tob
 		,rgtb.extra_base_hits_or_ebh
 from rolling_game_team_pitching rgtp
 join rolling_game_team_batting rgtb on rgtb.team_id=rgtp.team_id and rgtb.game_id=rgtp.game_id;

-- creating index on boxscore table and joined_team_batting_pitching to make the join faster
create or replace INDEX batting_pitching_game_index
ON joined_team_batting_pitching (game_id)
;

create or replace INDEX boxscore_game_index
ON boxscore (game_id)
;

CREATE or REPLACE TABLE joined_team_batting_pitching_boxscore
select bs.winner_home_or_away
 		,jtbp.*
from boxscore bs
join joined_team_batting_pitching jtbp_home
 on bs.game_id=jtbp_home.game_id AND jtbp_home.home_or_away='H';
join joined_team_batting_pitching jtbp_away
on bs.game_id=jtbp_away.game_id AND jtbp_away.home_or_away='A';

SELECT * FROM joined_team_batting_pitching_boxscore limit 1,100;

