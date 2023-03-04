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