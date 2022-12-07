create or replace table joined_game_team_pitching_counts
 select  g.local_date
                , g.game_id
                ,tbc.team_id
                ,tbc.Single
                ,tbc.Double
                ,tbc.Triple
                ,tbc.atBat
                ,tbc.Home_Run
                ,tbc.Hit
                ,tbc.Walk as base_on_balls
                ,tbc.plateApperance as plate_apperance
                ,tbc.Ground_Out as ground_out
                ,tbc.Flyout as fly_out
                ,tbc.Strikeout as strike_out
        from team_pitching_counts tbc
    join game g on tbc.game_id=g.game_id;
 select * from joined_game_team_pitching_counts limit 1,10;

 select t1.local_date as t1_date
        , t1.team_id
        , sum(t2.Single)
        , COUNT(*) AS cnt
    from joined_game_team_pitching_counts t1
    join joined_game_team_pitching_counts t2 ON
        t1.local_date > DATE_ADD(t2.local_date , INTERVAL -100 DAY) AND t1.team_id =t2.team_id
     group by t1.local_date, t1.team_id
     order by t1.team_id, t1.local_date ASC , t2.local_date ASC;


