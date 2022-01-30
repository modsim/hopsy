import anybadge

with open("docs/.docstrcovreport", "r") as f:
    docstr_coverage = f.readlines()[-1].split(": ")[1].strip('\n').strip('%')
    thresholds = {.5: 'red',
                  .6: 'orange',
                  .75: 'yellow',
                  .9: 'green'}

    badge = anybadge.Badge('docstr cov', str(docstr_coverage) + "%", thresholds=thresholds)
    badge.write_badge('docs/docstrcov.svg', overwrite=True)

