use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::batch::{read_manifest, CaseOutput};

const TREND_TOLERANCE_RELATIVE: f64 = 0.05;
const ABS_RATIO_REQUIREMENT: f64 = 0.95;
const TREND_CONSISTENCY_REQUIREMENT: f64 = 0.95;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseComparison {
    pub name: String,
    pub rust_time_us: f64,
    pub simai_time_us: Option<f64>,
    pub abs_ratio: Option<f64>,
    pub passes_abs_ratio: Option<bool>,
    pub schedule_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    pub cases: Vec<CaseComparison>,
    pub abs_ratio_limit: f64,
    pub abs_ratio_case_total: usize,
    pub abs_ratio_passed: usize,
    pub abs_ratio_failed: usize,
    pub abs_ratio_pass_rate: Option<f64>,
    pub abs_ratio_requirement: f64,
    pub passes_abs_ratio_requirement: Option<bool>,
    pub pairwise_total: usize,
    pub pairwise_consistent: usize,
    pub pairwise_inconsistent: usize,
    pub pairwise_rust_ties: usize,
    pub pairwise_consistency_rate: Option<f64>,
    pub inconsistent_pairs: Vec<PairwiseInconsistency>,
    pub unique_schedule_total: usize,
    pub unique_pairwise_total: usize,
    pub unique_pairwise_consistent: usize,
    pub unique_pairwise_inconsistent: usize,
    pub unique_pairwise_rust_ties: usize,
    pub unique_pairwise_consistency_rate: Option<f64>,
    pub unique_inconsistent_pairs: Vec<PairwiseInconsistency>,
    pub trend_tolerance_relative: f64,
    pub trend_consistency_requirement: f64,
    pub tolerant_unique_pairwise_total: usize,
    pub tolerant_unique_pairwise_consistent: usize,
    pub tolerant_unique_pairwise_inconsistent: usize,
    pub tolerant_unique_pairwise_rust_ties: usize,
    pub tolerant_unique_pairwise_consistency_rate: Option<f64>,
    pub passes_trend_requirement: Option<bool>,
    pub passes_alignment_requirement: Option<bool>,
    pub tolerant_unique_inconsistent_pairs: Vec<PairwiseInconsistency>,
    pub duplicate_schedule_groups: Vec<DuplicateScheduleGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseInconsistency {
    pub a: String,
    pub b: String,
    pub a_rust_time_us: f64,
    pub b_rust_time_us: f64,
    pub a_simai_time_us: f64,
    pub b_simai_time_us: f64,
    pub simai_delta_us: f64,
    pub rust_delta_us: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateScheduleGroup {
    pub schedule_hash: String,
    pub case_count: usize,
    pub names: Vec<String>,
    pub rust_time_us: f64,
    pub simai_min_us: Option<f64>,
    pub simai_max_us: Option<f64>,
    pub simai_spread_us: Option<f64>,
}

#[derive(Debug, Clone)]
struct UniqueSchedule {
    schedule_hash: String,
    names: Vec<String>,
    rust_time_us: f64,
    simai_time_us: f64,
}

pub fn compare_manifest(manifest_path: &Path, abs_ratio_limit: f64) -> Result<CompareReport> {
    let manifest = read_manifest(manifest_path)?;
    let mut cases = Vec::new();
    for case in &manifest.cases {
        let file = File::open(&case.rust_output)
            .with_context(|| format!("failed to open {}", case.rust_output.display()))?;
        let output: CaseOutput = serde_json::from_reader(file)
            .with_context(|| format!("failed to parse {}", case.rust_output.display()))?;
        let simai_time = match (case.simai_time_us, case.simai_end_to_end_csv.as_ref()) {
            (Some(time), _) => Some(time),
            (None, Some(csv)) => Some(read_simai_time_us(csv)?),
            (None, None) => None,
        };
        let abs_ratio = simai_time.map(|time| absolute_ratio(output.result.time_us, time));
        let schedule_hash = file_fingerprint(&case.translated)
            .with_context(|| format!("failed to hash {}", case.translated.display()))?;
        cases.push(CaseComparison {
            name: case.name.clone(),
            rust_time_us: output.result.time_us,
            simai_time_us: simai_time,
            abs_ratio,
            passes_abs_ratio: abs_ratio.map(|ratio| ratio <= abs_ratio_limit),
            schedule_hash,
        });
    }

    let pairwise = compare_pairs(cases.iter().filter_map(|case| {
        case.simai_time_us.map(|simai| PairPoint {
            name: case.name.clone(),
            rust_time_us: case.rust_time_us,
            simai_time_us: simai,
        })
    }));
    let duplicate_schedule_groups = duplicate_groups(&cases);
    let unique_schedules = unique_schedules(&cases);
    let unique_pairwise = compare_pairs(unique_schedules.iter().map(|schedule| PairPoint {
        name: format!(
            "{}:{}",
            &schedule.schedule_hash[..12.min(schedule.schedule_hash.len())],
            schedule.names.join("|")
        ),
        rust_time_us: schedule.rust_time_us,
        simai_time_us: schedule.simai_time_us,
    }));
    let tolerant_unique_pairwise = compare_pairs_with_tolerance(
        unique_schedules.iter().map(|schedule| PairPoint {
            name: format!(
                "{}:{}",
                &schedule.schedule_hash[..12.min(schedule.schedule_hash.len())],
                schedule.names.join("|")
            ),
            rust_time_us: schedule.rust_time_us,
            simai_time_us: schedule.simai_time_us,
        }),
        TREND_TOLERANCE_RELATIVE,
    );
    let abs_ratio_case_total = cases
        .iter()
        .filter(|case| case.passes_abs_ratio.is_some())
        .count();
    let abs_ratio_passed = cases
        .iter()
        .filter(|case| case.passes_abs_ratio == Some(true))
        .count();
    let abs_ratio_failed = cases
        .iter()
        .filter(|case| case.passes_abs_ratio == Some(false))
        .count();
    let abs_ratio_pass_rate = rate(abs_ratio_passed, abs_ratio_case_total);
    let passes_abs_ratio_requirement = if abs_ratio_case_total == 0 {
        None
    } else {
        abs_ratio_pass_rate.map(|rate| rate >= ABS_RATIO_REQUIREMENT)
    };
    let tolerant_unique_pairwise_consistency_rate = rate(
        tolerant_unique_pairwise.consistent,
        tolerant_unique_pairwise.total,
    );
    let passes_trend_requirement =
        tolerant_unique_pairwise_consistency_rate.map(|rate| rate >= TREND_CONSISTENCY_REQUIREMENT);
    let passes_alignment_requirement =
        match (passes_abs_ratio_requirement, passes_trend_requirement) {
            (Some(abs), Some(trend)) => Some(abs && trend),
            _ => None,
        };

    Ok(CompareReport {
        cases,
        abs_ratio_limit,
        abs_ratio_case_total,
        abs_ratio_passed,
        abs_ratio_failed,
        abs_ratio_pass_rate,
        abs_ratio_requirement: ABS_RATIO_REQUIREMENT,
        passes_abs_ratio_requirement,
        pairwise_total: pairwise.total,
        pairwise_consistent: pairwise.consistent,
        pairwise_inconsistent: pairwise.inconsistent,
        pairwise_rust_ties: pairwise.rust_ties,
        pairwise_consistency_rate: rate(pairwise.consistent, pairwise.total),
        inconsistent_pairs: pairwise.inconsistent_pairs,
        unique_schedule_total: unique_schedules.len(),
        unique_pairwise_total: unique_pairwise.total,
        unique_pairwise_consistent: unique_pairwise.consistent,
        unique_pairwise_inconsistent: unique_pairwise.inconsistent,
        unique_pairwise_rust_ties: unique_pairwise.rust_ties,
        unique_pairwise_consistency_rate: rate(unique_pairwise.consistent, unique_pairwise.total),
        unique_inconsistent_pairs: unique_pairwise.inconsistent_pairs,
        trend_tolerance_relative: TREND_TOLERANCE_RELATIVE,
        trend_consistency_requirement: TREND_CONSISTENCY_REQUIREMENT,
        tolerant_unique_pairwise_total: tolerant_unique_pairwise.total,
        tolerant_unique_pairwise_consistent: tolerant_unique_pairwise.consistent,
        tolerant_unique_pairwise_inconsistent: tolerant_unique_pairwise.inconsistent,
        tolerant_unique_pairwise_rust_ties: tolerant_unique_pairwise.rust_ties,
        tolerant_unique_pairwise_consistency_rate,
        passes_trend_requirement,
        passes_alignment_requirement,
        tolerant_unique_inconsistent_pairs: tolerant_unique_pairwise.inconsistent_pairs,
        duplicate_schedule_groups,
    })
}

pub fn write_compare_report(report: &CompareReport, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file =
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    serde_json::to_writer_pretty(file, report)?;
    Ok(())
}

fn absolute_ratio(a: f64, b: f64) -> f64 {
    a.max(b) / a.min(b)
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[derive(Debug, Clone)]
struct PairPoint {
    name: String,
    rust_time_us: f64,
    simai_time_us: f64,
}

#[derive(Debug, Default)]
struct PairStats {
    total: usize,
    consistent: usize,
    inconsistent: usize,
    rust_ties: usize,
    inconsistent_pairs: Vec<PairwiseInconsistency>,
}

fn compare_pairs<I>(points: I) -> PairStats
where
    I: IntoIterator<Item = PairPoint>,
{
    compare_pairs_with_tolerance(points, 0.0)
}

fn compare_pairs_with_tolerance<I>(points: I, simai_relative_tolerance: f64) -> PairStats
where
    I: IntoIterator<Item = PairPoint>,
{
    let points: Vec<_> = points.into_iter().collect();
    let mut stats = PairStats::default();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let a = &points[i];
            let b = &points[j];
            let simai_delta = a.simai_time_us - b.simai_time_us;
            let simai_tolerance_us =
                a.simai_time_us.min(b.simai_time_us) * simai_relative_tolerance;
            if simai_delta.abs() <= simai_tolerance_us {
                continue;
            }
            stats.total += 1;
            let rust_delta = a.rust_time_us - b.rust_time_us;
            if rust_delta.abs() < f64::EPSILON {
                stats.rust_ties += 1;
                stats.inconsistent += 1;
                stats.inconsistent_pairs.push(pair_inconsistency(a, b));
                continue;
            }
            let simai_order = a.simai_time_us.total_cmp(&b.simai_time_us);
            let rust_order = a.rust_time_us.total_cmp(&b.rust_time_us);
            if simai_order == rust_order {
                stats.consistent += 1;
            } else {
                stats.inconsistent += 1;
                stats.inconsistent_pairs.push(pair_inconsistency(a, b));
            }
        }
    }
    stats
}

fn pair_inconsistency(a: &PairPoint, b: &PairPoint) -> PairwiseInconsistency {
    PairwiseInconsistency {
        a: a.name.clone(),
        b: b.name.clone(),
        a_rust_time_us: a.rust_time_us,
        b_rust_time_us: b.rust_time_us,
        a_simai_time_us: a.simai_time_us,
        b_simai_time_us: b.simai_time_us,
        simai_delta_us: a.simai_time_us - b.simai_time_us,
        rust_delta_us: a.rust_time_us - b.rust_time_us,
    }
}

fn duplicate_groups(cases: &[CaseComparison]) -> Vec<DuplicateScheduleGroup> {
    let mut by_hash: BTreeMap<String, Vec<&CaseComparison>> = BTreeMap::new();
    for case in cases {
        by_hash
            .entry(case.schedule_hash.clone())
            .or_default()
            .push(case);
    }

    let mut groups = Vec::new();
    for (schedule_hash, group) in by_hash {
        if group.len() < 2 {
            continue;
        }
        let mut names: Vec<_> = group.iter().map(|case| case.name.clone()).collect();
        names.sort();
        let rust_time_us = group
            .iter()
            .map(|case| case.rust_time_us)
            .fold(f64::NEG_INFINITY, f64::max);
        let simai_times: Vec<_> = group.iter().filter_map(|case| case.simai_time_us).collect();
        let (simai_min_us, simai_max_us, simai_spread_us) = if simai_times.is_empty() {
            (None, None, None)
        } else {
            let min = simai_times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = simai_times
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            (Some(min), Some(max), Some(max - min))
        };
        groups.push(DuplicateScheduleGroup {
            schedule_hash,
            case_count: group.len(),
            names,
            rust_time_us,
            simai_min_us,
            simai_max_us,
            simai_spread_us,
        });
    }
    groups.sort_by(|a, b| {
        b.case_count.cmp(&a.case_count).then_with(|| {
            b.simai_spread_us
                .unwrap_or(0.0)
                .total_cmp(&a.simai_spread_us.unwrap_or(0.0))
        })
    });
    groups
}

fn unique_schedules(cases: &[CaseComparison]) -> Vec<UniqueSchedule> {
    let mut by_hash: BTreeMap<String, Vec<&CaseComparison>> = BTreeMap::new();
    for case in cases {
        if case.simai_time_us.is_some() {
            by_hash
                .entry(case.schedule_hash.clone())
                .or_default()
                .push(case);
        }
    }

    by_hash
        .into_iter()
        .map(|(schedule_hash, group)| {
            let mut names: Vec<_> = group.iter().map(|case| case.name.clone()).collect();
            names.sort();
            let len = group.len() as f64;
            let rust_time_us = group.iter().map(|case| case.rust_time_us).sum::<f64>() / len;
            let simai_time_us = group
                .iter()
                .filter_map(|case| case.simai_time_us)
                .sum::<f64>()
                / len;
            UniqueSchedule {
                schedule_hash,
                names,
                rust_time_us,
                simai_time_us,
            }
        })
        .collect()
}

fn file_fingerprint(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

fn read_simai_time_us(path: &Path) -> Result<f64> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read SimAI EndToEnd CSV {}", path.display()))?;
    let mut keyed = HashMap::new();
    let tokens: Vec<_> = text
        .lines()
        .flat_map(|line| line.split(','))
        .map(str::trim)
        .collect();
    for window in tokens.windows(2) {
        keyed.insert(window[0].to_string(), window[1].to_string());
    }
    if let Some(value) = keyed.get("workload finished at") {
        return value
            .parse::<f64>()
            .map_err(|err| anyhow!("invalid workload finished at value {}: {}", value, err));
    }
    if let Some(value) = keyed.get("total time") {
        return value
            .parse::<f64>()
            .map_err(|err| anyhow!("invalid total time value {}: {}", value, err));
    }
    Err(anyhow!(
        "could not find workload finished at or total time in {}",
        path.display()
    ))
}
