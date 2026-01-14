
export interface DiagnosisResponse {
  local_diagnosis: string;
  local_disease_type?: string;
  cloud_report: string;
  confidence?: number;
  verification_status: string;
  recommendations?: string;
  visualizations?: string[]; 
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}
