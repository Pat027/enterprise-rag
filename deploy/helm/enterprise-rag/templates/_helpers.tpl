{{/*
Expand the name of the chart.
*/}}
{{- define "enterprise-rag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name: <release>-<chart> (truncated to 63).
*/}}
{{- define "enterprise-rag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "enterprise-rag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "enterprise-rag.labels" -}}
helm.sh/chart: {{ include "enterprise-rag.chart" . }}
app.kubernetes.io/name: {{ include "enterprise-rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Component selector labels: pass dict with "ctx" and "component".
*/}}
{{- define "enterprise-rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "enterprise-rag.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Component labels (selector + common).
*/}}
{{- define "enterprise-rag.componentLabels" -}}
{{ include "enterprise-rag.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Resolve the secret name (existing or generated).
*/}}
{{- define "enterprise-rag.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "enterprise-rag.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "enterprise-rag.configMapName" -}}
{{- printf "%s-config" (include "enterprise-rag.fullname" .) -}}
{{- end -}}

{{/*
Standard envFrom block: configmap + secret.
*/}}
{{- define "enterprise-rag.envFrom" -}}
- configMapRef:
    name: {{ include "enterprise-rag.configMapName" . }}
- secretRef:
    name: {{ include "enterprise-rag.secretName" . }}
{{- end -}}

{{/*
Common env injecting HF_TOKEN as both HF_TOKEN and HUGGING_FACE_HUB_TOKEN
(vLLM image looks at the latter).
*/}}
{{- define "enterprise-rag.hfTokenEnv" -}}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "enterprise-rag.secretName" . }}
      key: HF_TOKEN
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "enterprise-rag.secretName" . }}
      key: HF_TOKEN
{{- end -}}
