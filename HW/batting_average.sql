create or replace table joined_game_team_pitching_counts
 select  		g.local_date
                ,g.game_id
                ,tbc.team_id
                ,tbc.Single
                ,tbc.Double
                ,tbc.Triple
                ,tbc.atBat
                ,tbc.Home_Run
                ,tbc.Hit
                ,tbc.Walk
                ,tbc.Ground_Out
                ,tbc.Flyout
                ,tbc.Strikeout
 from team_pitching_counts tbc
 join game g on tbc.game_id=g.game_id;

select * from joined_game_team_pitching_counts limit 1,100;

create or replace table rolling_game_team_pitching
select t1.local_date as t1_date
        , t1.game_id
        , t1.team_id
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
        , COUNT(*) AS cnt
    from joined_game_team_pitching_counts t1
    join joined_game_team_pitching_counts t2 ON
        t1.local_date > DATE_ADD(t2.local_date , INTERVAL -100 DAY) AND t1.team_id =t2.team_id
     group by t1.local_date, t1.team_id
     order by t1.team_id, t1.local_date ASC , t2.local_date ASC;
SELECT * FROM rolling_game_team_pitching limit 1,100;


