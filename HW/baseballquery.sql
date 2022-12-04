-- first join the game and batter counts table and save the results as a temporary table
create or replace temporary table joined_batter_game_bs as
    select
             g.local_date
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
            ,bs.winner_home_or_away
    from batter_counts bc
    join game g on bc.game_id=g.game_id
    join boxscore bs on g.game_id = bs.game_id
;

create or replace INDEX batter_indx
ON joined_batter_game_bs (batter,local_date)
;

create or replace temporary table last_100_days_per_player
    select    abg1.batter
            , abg1.local_date
            , abg1.Hit+sum(coalesce(abg2.Hit,0)) total_Hit
            , abg1.atBat+sum(coalesce(abg2.atBat,0)) total_atBat
            , abg1.Hit_By_Pitch+sum(coalesce(abg2.Hit_By_Pitch,0)) total_Hit_By_Pitch
            , abg1.Home_run+sum(coalesce(abg2.Home_run,0)) total_Home_run
            , abg1.Strikeout+sum(coalesce(abg2.Strikeout,0)) total_Strikeout
            , abg1.Single+sum(coalesce(abg2.Single,0)) total_Single
            , abg1.Double+sum(coalesce(abg2.Double,0)) total_Double
            , abg1.Triple+sum(coalesce(abg2.Triple,0)) total_Triple
            , abg1.Sac_Fly+sum(coalesce(abg2.Sac_Fly,0)) total_Sac_Fly
    from joined_batter_game_bs abg1 force index (batter_indx)
    left join joined_batter_game_bs abg2 on abg1.batter=abg2.batter
    and timestampdiff(day,abg1.local_date,abg2.local_date)<=99 and timestampdiff(day,abg1.local_date,abg2.local_date)>0
    group by abg1.batter,abg1.local_date,abg1.Hit, abg1.atBat, abg1.Hit_By_Pitch, abg1.Home_run, abg1.Strikeout,
    abg1.Single,abg1.Double, abg1.Triple, abg1.Sac_Fly
    order by 1,2
;
create or replace table last_100_days_rol_avg_per_player
    select
            last_100.batter
            ,sum(last_100.total_Hit)/sum(nullif(last_100.total_atBat,0)) as average_at_bat
            ,sum(last_100.total_Hit_By_Pitch)/sum(nullif(last_100.total_Hit_By_Pitch)) as average_hit_by_pitch
            ,sum(last_100.total_Home_run)/sum(nullif(last_100.total_Home_run)) as average_home_run
            ,sum(last_100.total_Strikeout)/sum(nullif(last_100.total_Strikeout)) as average_sum_Strikeout
            ,sum(last_100.total_Single)/sum(nullif(last_100.total_Single)) as average_single
            ,sum(last_100.total_Double)/sum(nullif(last_100.total_Double)) as average_Double
            ,sum(last_100.total_Triple)/sum(nullif(last_100.total_Triple)) as average_Triple
            ,sum(last_100.total_Sac_Fly)/sum(nullif(last_100.total_Sac_Fly)) as average_sac_fly

    from last_100_days_per_player last_100
    group by last_100.batter
;
select * from last_100_days_rol_avg_per_player limit 1,10;