create or replace table joined_batter_counts
 select  g.local_date
                , g.away_team_id
                , g.home_team_id
                ,bc.batter
                ,bc.game_id
                ,bc.Hit
                ,bc.atBat
                ,bc.team_id
                ,bc.homeTeam
                ,bc.awayTeam
                ,bc.Hit_By_Pitch
                ,bc.Home_run
                ,bc.Strikeout
                ,bc.Single
                ,bc.Double
                ,bc.Triple
                ,bc.Sac_Fly
        from batter_counts bc
    join game g on bc.game_id=g.game_id;


create or replace table historic_data
    select  jbg.batter
                ,jbg.team_id

            ,sum(jbg.Hit)/sum(nullif(jbg.atBat,0)) as Average_at_bat
            ,sum(jbg.Hit_By_Pitch) as sum_hit_by_pitch
            ,sum(jbg.Home_run) as sum_Home_run
            ,sum(jbg.Strikeout) as sum_Strikeout
            ,sum(jbg.Single) as sum_Single
            ,sum(jbg.Double) as sum_Double
            ,sum(jbg.Triple) as sum_Triple
            ,sum(jbg.Sac_Fly) as sum_Sac_Fly

    from joined_batter_counts_bs jbg
    group by jbg.batter
    order by jbg.batter ;

alter table historic_data add index team_indx (team_id)
;
create or replace table joined_historic_data_game
 select  bc.team_id,
        bc.game_id,
        hd.batter,
        hd.Average_at_bat
        ,hd.sum_hit_by_pitch
        ,hd.sum_Home_run
        ,hd.sum_Strikeout
        ,hd.sum_Single
        ,hd.sum_Double
        ,hd.sum_Triple
        ,hd.sum_Sac_Fly
        from historic_data hd
    join batter_counts bc on hd.team_id=bc.team_id;

create or replace table joined_historic_data_game_boxscore
     select
        hd.game_id
        ,hd.batter
        ,hd.Average_at_bat
        ,hd.sum_hit_by_pitch
        ,hd.sum_Home_run
        ,hd.sum_Strikeout
        ,hd.sum_Single
        ,hd.sum_Double
        ,hd.sum_Triple
        ,hd.sum_Sac_Fly
        ,bs.winner_home_or_away
        from joined_historic_data_game hd
    join boxscore bs on hd.game_id=bs.game_id;

create or replace table joined_historic_data_game_boxscore_added
     select
        hd.game_id
        , g.away_team_id
        , g.home_team_id
        ,hd.batter
        ,hd.Average_at_bat
        ,hd.sum_hit_by_pitch
        ,hd.sum_Home_run
        ,hd.sum_Strikeout
        ,hd.sum_Single
        ,hd.sum_Double
        ,hd.sum_Triple
        ,hd.sum_Sac_Fly
        from joined_historic_data_game_boxscore hd
    join game g on hd.game_id=g.game_id;


    select
        bc1.local_date as bc1_date
        , bc1.batter
        , sum(bc2.Hit)
        , COUNT(*) AS cnt
    from example bc1
    join example bc2 ON
        bc1.local_date > DATE_ADD(bc2.local_date , INTERVAL -100 DAY) AND bc1.batter =bc2.batter
    WHERE bc1.batter = 110029
     group by bc1.local_date, bc1.batter
     order by bc1.batter, bc1.local_date ASC , bc2.local_date ASC;


    select bc1.local_date, sum(bc2.Hit) as rolling_sum
    from example bc1 inner join example bc2 on datediff(bc1.local_date , bc2.local_date) between 0 and 99
     group by bc1.local_date
     order by bc1.local_date;