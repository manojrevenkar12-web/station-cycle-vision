# 4. Anonymise before anything leaves the box

Status: accepted

Video of people at work is personal data (GDPR) and, in Germany, needs the works council's agreement
(BetrVG §87(1) no. 6) when it can be used to monitor performance. The design keeps video on the box,
keeps only short clips around deviations for root-cause analysis, and pixelates every pixel the
person-mask head assigns to a person before the clip is written.

The threshold is low and the mask is dilated on purpose: a pixel of a person left visible is worse than
a pixelated pixel of bench. Leak rate (true person pixels left visible) is measured on every test set,
including people walking through the view, and is a release target (<= 1%). Cycle data is per
station, not per person; the events carry no identity.

This repository does not claim legal compliance; it shows the technical measures a DPIA would rely on.
