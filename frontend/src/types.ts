export interface SearchResult {
  doc_id: string;
  score: number;
  title: string;
  snippet: string;
  category: string;
  date: string;
  url: string;
}

export interface SearchMeta {
  query: string;
  total: number;
  page: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  start: number;
  end: number;
  sort_by: string;
  latency_ms: number;
}

export interface SearchParams {
  q: string;
  page?: number;
  sort?: string;
  category?: string;
  dateFrom?: string;
  dateTo?: string;
  fuzzy?: boolean;
  highlight?: boolean;
}

export interface Filters {
  category: string;
  sort: string;
  dateFrom: string;
  dateTo: string;
  fuzzy: boolean;
}

export interface AnalyticsSummary {
  total_searches: number;
  unique_queries: number;
  zero_result_rate: number;
  avg_results: number;
  avg_latency_ms: number;
}

export interface TopQuery {
  [key: string]: string | number;
  query: string;
  count: number;
  avg_results: number;
}

export interface VolumeEntry {
  date: string;
  count: number;
}

export interface AnalyticsData {
  summary: AnalyticsSummary;
  top: TopQuery[];
  zero: { query: string; count: number }[];
  volume: VolumeEntry[];
}
