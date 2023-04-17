---
title: 根据select进行update
date: 2023-02-21 16:15:42
tags:
---
```sql
update
  LabelComInvest
set
  lc_code = replace(b.lc_code, '	', '')
from
   LabelComInvest a,
  (select * from LabelComInvest)b
where
   a.lc_id=b.lc_id
```

```sql
declare @span int
declare @end int
declare @start int
declare @log varchar(1000)
set @span = 500000		-- 50万搜索跨度
set @end = 263802217	--
set @start = @end - @span

while (@end > 0)
begin

set @log = '当前处理范围：' + CAST(@start AS VARCHAR)+ '-'+ CAST(@end AS VARCHAR)
RAISERROR (@log, 10, 1) WITH NOWAIT

-- step1: 将 od_faren 从'-' 变成 ''，插入到 dtl 同步表
insert into [QZOrgCompanySync].[dbo].[OrgCompanyDtlSyncTemp] (od_oc_code
    ,od_faRen 
	,od_regM ,od_regMoney ,od_regDate
    ,od_chkDate  ,od_regType,od_bussinessS ,od_bussinessDes , od_ext
    ,od_status ,od_CreateTime ,od_source ,od_yearChk   ,od_bussinessE
    ,od_factMoney  ,od_factM  ,od_syncStatus  ,od_oc_area
    ,od_oc_areaName)
values 
select 
      [od_oc_code]
      ,od_faRen
	  ,[od_regM] , 
case  when od_regMoney like '%人民币元%' then REPLACE(od_regMoney, '人民币元', '万元人民币') else REPLACE( REPLACE(od_regMoney, '元', '万元人民币'), '人民币', '万元人民币') end -- od_regMoney 加 万
 ,[od_regDate]     
      ,[od_chkDate] ,[od_regType] ,[od_bussinessS] ,[od_bussinessDes],[od_ext] 
      ,[od_status] ,[od_CreateTime] ,[od_source] ,[od_yearChk], [od_bussinessE]
	  , od_factMoney, od_factM, '', '', ''
 from [QZOrgCompany].[dbo].[OrgCompanyDtl] with (nolock) where od_regMoney not like '%万%' and od_regMoney like '%人民币%' and od_id > @start and od_id <= @end

 -- step2: 将 dtl 表的 od_faren 从 '-' 变成 ''
update [QZOrgCompany].[dbo].[OrgCompanyDtl] WITH (ROWLOCK) 
set od_regMoney = case  when od_regMoney like '%人民币元%' then REPLACE(od_regMoney, '人民币元', '万元人民币') else REPLACE( REPLACE(od_regMoney, '元', '万元人民币'), '人民币', '万元人民币') end
where od_regMoney not like '%万%'and od_regMoney like '%人民币%' and od_id > @start and od_id <= @end

set @end = @start
set @start = @start - @span

end

```