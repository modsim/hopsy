import anybadge

with open("docs/.docstrcovreport.txt", "r") as f:
    docstr_coverage = int(float(f.readlines()[-1].split(": ")[1].strip('\n').strip('%')))
    thresholds = {50: 'red',
                  60: 'orange',
                  75: 'yellow',
                  90: 'green'}

    badge = anybadge.Badge('docstr cov', docstr_coverage, value_suffix="%", thresholds=thresholds)
    badge.write_badge('docs/docstrcov.svg', overwrite=True)

