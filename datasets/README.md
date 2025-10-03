# Datasets

## Datasets Description

### QiLin, DanYangXinQu, XueBu

**Source**  
Electronic Toll Collection (ETC) gantry records collected from the QiLin Interchange, a major cloverleaf hub in Jiangsu Province, China.  

**Description**  
The QiLin dataset is constructed from multi-source ETC transaction data, which records vehicle passages at mainline and ramp gantries.  
For each turning movement within the interchange, traffic flow is paired with contextual features including:  

- Ramp volume
- Upstream and downstream gantry volume and speed  
- Mainline geometry attributes (width, lane number)  
- Ramp lane number  

After cleaning and aligning the raw records, the data is aggregated into fixed intervals to form a multivariate time series suitable for spatio-temporal traffic forecasting.  

**Period**  
Continuous multi-week ETC data (exact collection period depending on the experimental setup).  

**Granularity**  
Configurable; in this study, data is sampled at **3-minute interval**, **5-minute interval**, **10-minute interval**.  

**Number of Time Steps**  
144 steps per day (at 10-minute resolution), totaling approximately 23 days.  

**Dataset Split**  
17:3:3 (train/validation/test).  

**Variates**  
- **Turning volume = Ramp volume (target flow)**  
- Upstream gantry volume and speed  
- Downstream gantry volume and speed  
- Upstream mainline width and lane number  
- Downstream mainline width and lane number  
- Ramp lane number  
- Temporal features: time of day, day of week  

**Typical Settings**  
- Input length: 12 steps
- Output length: 12 steps
- Forecasting task: ramp turning flow prediction under detector-scarce conditions  