# Competitive Intelligence Report: {{ report.company_name }}
**Report Date:** {{ report.report_date }}

## Executive Summary
{% for point in report.executive_summary %}
- {{ point }}
{% endfor %}

## Competitor Analysis
{% if report.competitors %}
{% for competitor in report.competitors %}
### {{ competitor.name }}
**Company Profile:**
- Type: {{ competitor.profile.company_type }}
- HQ: {{ competitor.profile.headquarters }}
- Founded: {{ competitor.profile.founded }}
- Employees: {{ competitor.profile.employees }}
- Revenue: {{ competitor.profile.revenue }}

**Market Position:**
- Global Rank: {{ competitor.market_position.global_rank }}
- Market Share: {{ competitor.market_position.market_share }}
- Target Segments: {{ competitor.market_position.target_segments | join(', ') }}
- Geographic Presence: {{ competitor.market_position.geographic_presence | join(', ') }}

**Recent Activities:**
{% for activity in competitor.recent_activities %}
- {{ activity.date }}: {{ activity.description }} (Impact: {{ activity.impact }})
{% endfor %}

**Products:**
{% for product in competitor.products %}
- {{ product.name }} ({{ product.category }}): {{ product.description }}
{% endfor %}

**SWOT Analysis:**
- Strengths: {{ competitor.swot.strengths | join(', ') }}
- Weaknesses: {{ competitor.swot.weaknesses | join(', ') }}
- Opportunities: {{ competitor.swot.opportunities | join(', ') }}
- Threats: {{ competitor.swot.threats | join(', ') }}

**Financial Metrics:**
- Revenue Growth: {{ competitor.financials.revenue_growth }}
- Market Cap: {{ competitor.financials.market_cap }}
- R&D Investment: {{ competitor.financials.r_and_d_investment }}

{% endfor %}
{% endif %}

## Market News
{% if report.market_news %}
{% for news in report.market_news %}
### {{ news.title }}
- Date: {{ news.timestamp }}
- Company: {{ news.company }}
- Source: {{ news.source }}
- Summary: {{ news.summary }}

**Impact Analysis:**
- Market Impact: {{ news.impact_analysis.market_impact }}
- Competitive Implications: {{ news.impact_analysis.competitive_implications }}
- Industry Trends: {{ news.impact_analysis.industry_trends | join(', ') }}

Categories: {{ news.categories | join(', ') }}

{% endfor %}
{% endif %}

## Market Trends
{% if report.market_trends %}
{% for trend in report.market_trends %}
### {{ trend.name }} ({{ trend.category }})
{{ trend.description }}

**Drivers:**
{% for driver in trend.drivers %}
- {{ driver }}
{% endfor %}

**Impact Assessment:**
- Market Impact: {{ trend.impact_assessment.market_impact }}
- Timeframe: {{ trend.impact_assessment.timeframe }}
- Confidence Level: {{ trend.impact_assessment.confidence_level }}

**Affected Companies:** {{ trend.affected_companies | join(', ') }}

**Recommendations:**
{% for rec in trend.recommendations %}
- {{ rec }}
{% endfor %}

{% endfor %}
{% endif %}

## Strategic Recommendations
{% for rec in report.strategic_recommendations %}
### {{ rec.action }}
- Priority: {{ rec.priority }}
- Expected Impact: {{ rec.expected_impact }}
- Timeline: {{ rec.timeline }}
- Resources Needed: {{ rec.resources_needed | join(', ') }}

{% endfor %}

## Risk Assessment
{% for risk in report.risk_assessment %}
### {{ risk.risk_type }}
- Description: {{ risk.description }}
- Severity: {{ risk.severity }}
- Likelihood: {{ risk.likelihood }}

**Mitigation Steps:**
{% for step in risk.mitigation_steps %}
- {{ step }}
{% endfor %}

{% endfor %}

## Opportunities
{% for opp in report.opportunities %}
### {{ opp.name }}
- Description: {{ opp.description }}
- Potential Impact: {{ opp.potential_impact }}
- Timeline: {{ opp.timeline }}

**Requirements:**
{% for req in opp.requirements %}
- {{ req }}
{% endfor %}

{% endfor %}

## Monitoring Priorities
{% for priority in report.monitoring_priorities %}
- {{ priority }}
{% endfor %}
