
-- first join the game and batter counts table and save the results as a temporary table
create or replace temporary table joined_batter_game as
    select
            g.local_date
            ,bc.batter
            ,bc.game_id
            ,bc.Hit
            ,bc.atBat
    from batter_counts bc
    join game g on bc.game_id=g.game_id
;

-- calculate annual batting average
create or replace temporary table annual_at_bat_per_player
    select
            jbg.batter
            ,year(jbg.local_date) as date_year
            ,sum(jbg.Hit)/sum(nullif(jbg.atBat,0)) as Average_at_bat
    from joined_batter_game jbg
    group by jbg.batter, date_year
    order by date_year
;

-- calculate historic batting average
create or replace temporary table historic_at_bat_per_player
    select
            jbg.batter
            ,sum(jbg.Hit)/sum(nullif(jbg.atBat,0)) as Average_at_bat
    from joined_batter_game jbg
    group by jbg.batter
;

-- calculate last 100 days batting average --
-- first we create a table with the 99 recent entries for each player --

alter table joined_batter_game add index batter_indx (batter,local_date)
;

create or replace INDEX batter_indx
ON joined_batter_game (batter,local_date)
;


create or replace temporary table last_100_days_per_player
    select    abg1.batter
            , abg1.local_date
            , abg1.Hit+sum(coalesce(abg2.Hit,0)) total_Hit
            , abg1.atBat+sum(coalesce(abg2.atBat,0)) total_atBat
    from joined_batter_game abg1 force index (batter_indx)
    left join joined_batter_game abg2 on abg1.batter=abg2.batter
    and timestampdiff(day,abg1.local_date,abg2.local_date)<=99 and timestampdiff(day,abg1.local_date,abg2.local_date)>0
    group by abg1.batter,abg1.local_date,abg1.Hit, abg1.atBat
    order by 1,2
;

create or replace table last_100_days_rol_avg_per_player
    select
            last_100.batter
            ,sum(last_100.total_Hit)/sum(nullif(last_100.total_atBat,0)) as Average_at_bat
    from last_100_days_per_player last_100
    group by last_100.batter
;
select * from last_100_days_rol_avg_per_player limit 1,10;
