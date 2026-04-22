# Pi0-Model-Note  

## Inference Tracing  

- Run VLM part for KV-cache  
    - Locate at [pi0.py:237](src/openpi/models/pi0.py#L237)  
- At each denoise step, not inferencing VLM (use KV-cache) by inputting `None` as embedding at [gemma.py:392](src/openpi/models/gemma.py#L392).  

## Decoded Text from Hidden States of the Prefix  

```text
---
data: RegistryLitePullParserCELONAastéroïdesPullParserPullParser parutionRegistryLite
---
data: RegistryLitePullParserCELONASceneManagementPullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAastéroïdesPullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAastéroïdesPullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONASceneManagementPullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONASceneManagementastéroïdesPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONASceneManagementastéroïdesPullParser parutionRegistryLite
---
data: RegistryLite سكانيةCELONAReferencieastéroïdesPullParser parutionRegistryLite
---
data: RegistryLite سكانيةCELONAReferencieastéroïdesPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAReferencieastéroïdesPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAReferenciePullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAReferenciePullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAReferenciePullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAReferenciePullParserPullParser parutionRegistryLite
---
data: RegistryLite BoxFitCELONAReferenciePullParserPullParser parutionRegistryLite
---
data: RegistryLite سكانيةCELONAReferencie سكانيةastéroïdes parutionRegistryLite
---
data: RegistryLite سكانيةCELONAReferenciePullParserastéroïdes parutionRegistryLite
---
data: 葩 سكانيةCELONAŽivot سكانيةastéroïdes Longueur ویکی‌آمباردا
---
data: 葩 سكانيةCELONAReferencieastéroïdesastéroïdes parutionRegistryLite
---
data: 葩 سكانيةCELONAŽivotastéroïdesastéroïdes Longueur ویکی‌آمباردا
---
data: 葩 سكانيةCELONAReferencieastéroïdesastéroïdes parution ویکی‌آمباردا
---
data: 葩 سكانيةCELONAŽivot سكانيةastéroïdes LongueurContentAsync
---
data: 葩 سكانيةastéroïdesLiteratastéroïdesbatore conformitéBrainz
---
data: 葩 سكانيةbatoreSceneManagementastéroïdesastéroïdes parutionContentAsync
---
data: 葩 سكانيةbatoreExternéUpInsideSceneManagement parutionContentAsync
---
data: 葩 سكانيةbatoreExternéUpInsideSceneManagement parution ویکی‌آمباردا
---
```
