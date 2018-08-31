# Global Burden of Disease from IHME

Source: [GBD 2016](http://ghdx.healthdata.org/gbd-2016). The source
files are downloaded from [GBD result tool](http://ghdx.healthdata.org/gbd-results-tool)

# Indicators in this dataset

Below measures are available for the context "cause" in GBD:

- deaths

And each measure is available in 3 units (except MMR):

- Number
- Rate per 100K people
- Percent

We have combined these measures and metrics, and created following
indicators in this dataset:

- deaths_number
- deaths_percent
- deaths_rate

## Notes

1. uncertainty level upper bound and lower bound values are not
   included in the dataset. Only median values are included.
2. at most 2 decimal digits are kept for numbers.
3. only country level data available; city/state/region/world level
   data are not in the dataset
4. all specific age groups are avaliable; for aggregated age groups
   "all age" and "age standardized" are available
