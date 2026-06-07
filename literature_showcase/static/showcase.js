const dataEl = document.querySelector("#showcaseData");
const reproductionEl = document.querySelector("#reproductionData");
const initialEl = document.querySelector("#initialView");
const activeRunEl = document.querySelector("#activeRunId");
const queryMetaEl = document.querySelector("#queryMetaData");
const reviewData = window.SHOWCASE_DATA || JSON.parse(dataEl?.textContent || "{}");
const reproductionData = JSON.parse(reproductionEl?.textContent || '{"targets":[]}');
const initialView = JSON.parse(initialEl?.textContent || '{"layer":"overview"}');
const activeRunId = JSON.parse(activeRunEl?.textContent || '""');
const queryMetaData = JSON.parse(queryMetaEl?.textContent || "{}");
const urlParams = new URLSearchParams(window.location.search);

const canvas = document.querySelector("#reviewCanvas");
const pageTitle = document.querySelector("#pageTitle");
const pageSubtitle = document.querySelector("#pageSubtitle");
const queryMetaBox = document.querySelector("#queryMeta");
const layerEyebrow = document.querySelector("#layerEyebrow");
const breadcrumb = document.querySelector("#breadcrumb");
const backButton = document.querySelector("#backButton");
const overviewButton = document.querySelector("#overviewButton");
const literatureViewButton = document.querySelector("#literatureViewButton");
const reproductionViewButton = document.querySelector("#reproductionViewButton");
const startButton = document.querySelector("#startJob");
const state = document.querySelector("#jobState");
const addTopicClause = document.querySelector("#addTopicClause");
const topicClauses = document.querySelector("#topicClauses");
const progressPanel = document.querySelector("#jobProgress");
const progressStage = document.querySelector("#progressStage");
const progressRunId = document.querySelector("#progressRunId");
const progressPercent = document.querySelector("#progressPercent");
const progressBar = document.querySelector("#progressBar");
const progressSearched = document.querySelector("#progressSearched");
const progressFiltered = document.querySelector("#progressFiltered");
const progressDownloaded = document.querySelector("#progressDownloaded");
const progressAnalyzed = document.querySelector("#progressAnalyzed");
const progressSteps = document.querySelector("#progressSteps");
const progressNotes = document.querySelector("#progressNotes");
const JOB_STORE_KEY = "literatureShowcaseJobs";

let currentDirection = null;
let currentPaper = null;
let activePollTimer = null;
let activeReproPollTimer = null;
let visibleJobId = "";
const missingJobRetries = {};
const reproChatHistories = {};

function isTerminalJobStatus(status) {
  return ["completed", "failed"].includes(String(status || ""));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderQueryMeta() {
  if (!queryMetaBox) return;
  const conditions = Array.isArray(queryMetaData.conditions) ? queryMetaData.conditions : [];
  const topic = queryMetaData.topic || reviewData.topic || "";
  if (!topic && !conditions.length) {
    queryMetaBox.classList.add("hidden");
    return;
  }
  const rows = [];
  if (topic) {
    rows.push(`<span class="query-topic"><b>主题</b>${escapeHtml(topic)}</span>`);
  }
  const topicExpanded = (Array.isArray(queryMetaData.topic_expanded_keywords) ? queryMetaData.topic_expanded_keywords : []).filter(Boolean);
  if (topicExpanded.length) {
    rows.push(`
      <span class="query-topic-expanded">
        <b>主题拓展</b>
        <small>${topicExpanded.map((item) => `<i>${escapeHtml(item)}</i>`).join("")}</small>
      </span>
    `);
  }
  if (queryMetaData.year_from || queryMetaData.year_to) {
    rows.push(`<span class="query-years"><b>年份</b>${escapeHtml(queryMetaData.year_from || "不限")} - ${escapeHtml(queryMetaData.year_to || "不限")}</span>`);
  }
  for (const condition of conditions) {
    const inputs = (condition.input_keywords || []).filter(Boolean);
    const expanded = (condition.expanded_keywords || []).filter(Boolean);
    if (!inputs.length && !expanded.length) continue;
    const logicLabel = condition.logic ? `${condition.logic}条件` : "条件";
    const logicClass = String(condition.logic_raw || condition.logic || "").toLowerCase();
    rows.push(`
      <span class="query-condition query-condition-${escapeHtml(logicClass)}">
        <b>${escapeHtml(logicLabel)}</b>
        ${inputs.map((item) => `<strong>${escapeHtml(item)}</strong>`).join("")}
        ${expanded.length ? `<small>拓展：${expanded.map((item) => `<i>${escapeHtml(item)}</i>`).join("")}</small>` : ""}
      </span>
    `);
  }
  queryMetaBox.innerHTML = rows.join("");
  queryMetaBox.classList.remove("hidden");
}

function directionUrl(id) {
  const url = `/direction/${encodeURIComponent(id)}`;
  return activeRunId ? `${url}?run=${encodeURIComponent(activeRunId)}` : url;
}

function paperUrl(directionId, paperId) {
  const url = `/paper/${encodeURIComponent(directionId)}/${encodeURIComponent(paperId)}`;
  return activeRunId ? `${url}?run=${encodeURIComponent(activeRunId)}` : url;
}

function overviewUrl() {
  return activeRunId ? `/?run=${encodeURIComponent(activeRunId)}` : "/";
}

if (overviewButton) {
  overviewButton.href = overviewUrl();
}

function getDirection(id) {
  return (reviewData.directions || []).find((item) => item.id === id);
}

function getPaper(direction, paperId) {
  return (direction?.papers || []).find((item) => item.id === paperId);
}

function findPaperLocation(paperId) {
  for (const direction of reviewData.directions || []) {
    const paper = getPaper(direction, paperId);
    if (paper) return {direction, paper};
  }
  return null;
}

function tags(items) {
  return `<div class="tag-list">${(items || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function inlineMarkdown(text) {
  return typesetInlineLatex(
    escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
  );
}

function normalizePlainFormulaPhrases(text) {
  return String(text ?? "").replace(
    /V_dc_ref\s*=\s*sqrt\(V_dc0\^2\s*\+\s*\(2H_em S_n\/\(NC ω0\)\)\(ω-ω0\)\)/g,
    "\\(V_{dc}^{ref} = \\sqrt{V_{dc0}^{2} + \\frac{2H_{em}S_n}{NC\\omega_0}(\\omega-\\omega_0)}\\)"
  );
}

function typesetInlineLatex(html) {
  return String(html ?? "")
    .split(/(<[^>]+>)/g)
    .map((part) => {
      if (!part || part.startsWith("<")) return part;
      return part
        .split(/(\\\(.+?\\\)|\\\[.+?\\\]|\$\$.+?\$\$|\$(?!\s).+?(?<!\s)\$)/g)
        .map((segment) => {
          if (!segment) return segment;
          if (/^\\\(.+\\\)$/.test(segment)) {
            return `\\(${normalizeInlineLatexSource(segment.slice(2, -2))}\\)`;
          }
          if (/^\\\[.+\\\]$/.test(segment)) {
            return `\\[${normalizeInlineLatexSource(segment.slice(2, -2))}\\]`;
          }
          if (/^\$\$.+\$\$$/.test(segment)) {
            return `\\[${normalizeInlineLatexSource(segment.slice(2, -2))}\\]`;
          }
          if (/^\$(?!\$).+\$$/.test(segment)) {
            return `\\(${normalizeInlineLatexSource(segment.slice(1, -1))}\\)`;
          }
          const normalizedPhrase = normalizePlainFormulaPhrases(segment);
          if (normalizedPhrase !== segment) return normalizedPhrase;
          return segment
            .replace(/([A-Za-zΑ-Ωα-ω]+(?:\([^()（）]{1,32}\))?\s*=\s*[0-9A-Za-zΑ-Ωα-ω.\\+\-*/^_{}()[\],，\s]{2,80})/g, (_, source) => `\\(${normalizeInlineLatexSource(source.trim())}\\)`)
            .replace(/(^|[\s(（,，;；:：])((?:[A-Za-z][A-Za-z0-9]*)(?:_\{?[A-Za-z0-9]+\}?|\^\{?[A-Za-z0-9]+\}?)+(?:_\{?[A-Za-z0-9]+\}?|\^\{?[A-Za-z0-9]+\}?)*)/g, (_, prefix, source) => `${prefix}\\(${normalizeInlineLatexSource(source)}\\)`);
        })
        .join("");
    })
    .join("");
}

function normalizeLatexSource(source) {
  return String(source ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\\text(?!\s*\{)([A-Za-z0-9_-]+)/g, "\\text{$1}")
    .replace(/([A-Za-z]+)\\_([A-Za-z][A-Za-z0-9]*)_\{([^{}]+)\}/g, "$1_{\\text{$2},$3}")
    .replace(/([A-Za-z]+)\\_([A-Za-z][A-Za-z0-9]*)/g, "$1_{\\text{$2}}")
    .replace(/\\_/g, ",")
    .replace(/([A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ω]*)_\\([A-Za-z]+)/g, "$1_{\\$2}")
    .replace(/([A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ω]*)\^\\([A-Za-z]+)/g, "$1^{\\$2}");
}

function normalizeInlineLatexSource(source) {
  return normalizeLatexSource(source)
    .replace(/\s+/g, " ")
    .replace(/\b([A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ω]*(?:_[A-Za-z0-9Α-Ωα-ω]+){2,})\b/g, (token) => {
      const parts = token.split("_");
      return `${parts[0]}_{${parts.slice(1).join(",")}}`;
    })
    .replace(/([A-Za-zΑ-Ωα-ω])_([A-Za-z0-9Α-Ωα-ω]+)(?![A-Za-z0-9Α-Ωα-ω])/g, "$1_{$2}")
    .replace(/([A-Za-zΑ-Ωα-ω])\^([0-9]+(?:\.[0-9]+)?|[A-Za-zΑ-Ωα-ω]+)/g, "$1^{$2}");
}

function renderMarkdown(value) {
  const source = String(value ?? "").replace(/\r\n/g, "\n").trim();
  if (!source) return "";
  const blocks = source.split(/\n{2,}/);
  return blocks.map((block) => {
    const trimmed = block.trim();
    if (!trimmed) return "";
    if ((trimmed.startsWith("$$") && trimmed.endsWith("$$")) || (trimmed.startsWith("\\[") && trimmed.endsWith("\\]"))) {
      return `<div class="math-block">${escapeHtml(trimmed)}</div>`;
    }
    if (/^#{1,4}\s+/.test(trimmed)) {
      const level = Math.min((trimmed.match(/^#+/) || [""])[0].length + 3, 6);
      return `<h${level} class="markdown-heading">${inlineMarkdown(trimmed.replace(/^#{1,4}\s+/, ""))}</h${level}>`;
    }
    if (/^[-*]\s+/m.test(trimmed)) {
      const items = trimmed.split("\n").filter(Boolean).map((line) => `<li>${inlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>`).join("");
      return `<ul class="markdown-list">${items}</ul>`;
    }
    return `<p>${inlineMarkdown(trimmed).replace(/\n/g, "<br>")}</p>`;
  }).join("");
}

function markdownBlock(value, className = "summary-text") {
  return `<div class="${className}">${renderMarkdown(value)}</div>`;
}

function renderCorpusOverview(corpus) {
  const topic = cleanText(corpus.topic || reviewData.topic || "当前主题");
  const paperTotal = Number(corpus.paper_total || 0);
  const targetTotal = Number(corpus.target_paper_total || 0);
  const directionTotal = Number(corpus.direction_total || (reviewData.directions || []).length || 0);
  const yearRange = cleanText(corpus.year_range);
  const methods = asArray(corpus.methods).map((item) => cleanText(item)).filter(Boolean).slice(0, 6);
  const insights = asArray(corpus.domain_insights)
    .map((item) => cleanText(item?.summary || item?.title || item))
    .filter(Boolean)
    .slice(0, 3);
  const opportunities = asArray(corpus.future_opportunities || corpus.gap)
    .map((item) => cleanText(item))
    .filter(Boolean)
    .slice(0, 3);

  const scopeParts = [];
  if (paperTotal) scopeParts.push(`${paperTotal} 篇文献`);
  if (targetTotal && targetTotal > paperTotal) scopeParts.push(`目标样本 ${targetTotal} 篇`);
  if (directionTotal) scopeParts.push(`${directionTotal} 个研究方向`);
  if (yearRange && yearRange !== "-") scopeParts.push(`时间范围 ${yearRange}`);
  const scopeText = scopeParts.length
    ? `本次综述围绕“${topic}”整理了${scopeParts.join("、")}，用于从总体问题、方向划分、方法谱系和未来机会四个层次把握领域版图。`
    : `本次综述围绕“${topic}”组织文献，重点呈现研究对象、方法路径、结论共识和待解决问题。`;

  const directionNames = (reviewData.directions || []).slice(0, 5).map((direction) => `${direction.id} ${direction.name}`).join("、");
  const items = [compactText(scopeText, 115)];
  if (directionNames) {
    items.push(compactText(`方向结构包括${directionNames}，分别对应价值评估、竞价运行、投资配置、聚合参与或边界背景等不同问题。`, 115));
  }
  if (methods.length) {
    items.push(compactText(`方法上主要覆盖${methods.join("、")}等路径，可比较建模对象、约束处理、求解方式和评价指标。`, 95));
  }
  if (insights.length || opportunities.length) {
    const insightText = insights.length ? `共识：${insights.map((item) => compactText(item, 45)).join("；")}。` : "";
    const opportunityText = opportunities.length ? `机会：${opportunities.map((item) => compactText(item, 42)).join("；")}。` : "";
    items.push(compactText(`${insightText}${opportunityText}`, 110));
  }

  return `
    <div class="summary-text overview-rich">
      <ul class="point-list overview-points">
        ${items.slice(0, 4).map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function cleanText(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .replace(/…|\.{3,}|。{3,}/g, "。")
    .replace(/([。！？；;，,、])\1+/g, "$1")
    .replace(/。+[；;]/g, "；")
    .replace(/[；;]+。/g, "。")
    .replace(/，。/g, "。")
    .replace(/、。/g, "。")
    .replace(/([。！？；;])([，,、])/g, "$1")
    .trim();
}

function compactText(value, limit = 96) {
  const text = cleanText(value);
  if (text.length <= limit) return text;
  const boundary = Math.max(
    text.lastIndexOf("。", limit),
    text.lastIndexOf("；", limit),
    text.lastIndexOf("，", limit),
    text.lastIndexOf(".", limit),
    text.lastIndexOf(";", limit),
    text.lastIndexOf(",", limit)
  );
  const cut = boundary > 28 ? boundary + 1 : limit;
  return cleanText(text.slice(0, cut).replace(/[，,；;、。]*$/, "。"));
}

function pointItems(value, maxItems = 3, limit = 200) {
  const text = cleanText(value);
  if (!text) return ["暂无明确信息"];
  let items = text
    .split(/(?:[。；;]\s*|\n+|(?:^|\s)\d+[.)、]\s*)/)
    .map((item) => cleanText(item))
    .filter(Boolean);
  if (items.length <= 1 && text.length > limit) {
    items = text
      .split(/[，,、]\s*/)
      .map((item) => cleanText(item))
      .filter(Boolean);
  }
  return items.slice(0, maxItems).map((item) => compactText(item, limit));
}

function pointList(value, maxItems = 3, limit = 200) {
  return `<ul class="point-list">${pointItems(value, maxItems, limit).map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`;
}

function asArray(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function paperOriginalUrl(paper) {
  const url = cleanText(paper?.url);
  const doi = cleanText(paper?.doi);
  if (/^https?:\/\//i.test(url)) return url;
  if (/^doi:/i.test(url)) return `https://doi.org/${encodeURI(url.replace(/^doi:/i, "").trim())}`;
  if (/^10\.\S+/i.test(url)) return `https://doi.org/${encodeURI(url)}`;
  if (/^https?:\/\//i.test(doi)) return doi;
  if (doi) return `https://doi.org/${encodeURI(doi.replace(/^doi:/i, "").trim())}`;
  return "";
}

function renderOriginalLinkMeta(paper) {
  const url = paperOriginalUrl(paper);
  if (!url) return "";
  return `<div class="paper-source-row"><span>原文</span><b><a href="${escapeHtml(url)}" target="_blank" rel="noopener">打开原文链接</a></b></div>`;
}

function sectionPointList(value, maxItems = 3, limit = 300) {
  return pointList(value, maxItems, limit);
}

function renderInsightTimeline(corpus) {
  const directions = reviewData.directions || [];
  const source = directions.length ? directions.map((direction, index) => ({
    period: direction.id || `D${index + 1}`,
    theme: direction.name || "研究方向",
    description: cleanText(`${direction.summary || direction.core_question || ""} 该方向主要方法包括 ${(direction.methods_distribution || []).slice(0, 3).map((item) => item.method).join("、") || "待进一步提炼"}。`)
  })) : (corpus.timeline || []).map((item) => ({
    period: item.period,
    theme: item.theme,
    description: item.description
  }));
  return `
    <div class="timeline">
      ${source.slice(0, 6).map((item) => `
        <article>
          <b>${escapeHtml(item.period)} · ${escapeHtml(item.theme)}</b>
          <p>${escapeHtml(compactText(item.description, 180))}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function evidencePaperLinks(paperIds) {
  const links = (paperIds || []).map((paperId) => {
    const location = findPaperLocation(paperId);
    if (!location) return `<span>${escapeHtml(paperId)}</span>`;
    return `<a href="${paperUrl(location.direction.id, paperId)}">${escapeHtml(paperDisplayTitle(location.paper))}</a>`;
  });
  return links.length ? links.join("") : "暂无对应文献";
}

function representativePaperIds(direction, limit = 3) {
  return (direction.papers || [])
    .filter((paper) => paper && paper.id)
    .slice(0, limit)
    .map((paper) => paper.id);
}

function domainInsightRows(corpus) {
  const explicitRows = asArray(corpus.domain_insights).filter((item) => item && cleanText(item.title || item.summary || item.explanation));
  if (explicitRows.length) {
    return explicitRows.slice(0, 6).map((item) => ({
      title: item.title || "领域洞察",
      summary: item.summary || item.explanation || item.claim,
      directions: asArray(item.support_directions || item.directions).filter(Boolean),
      paperCount: item.support_paper_count || asArray(item.papers).length,
      papers: asArray(item.papers).filter(Boolean)
    }));
  }
  return (reviewData.directions || []).slice(0, 6).map((direction) => ({
    title: `${direction.name || direction.id} 是当前主题下的关键研究方向`,
    summary: direction.summary || direction.core_question || `${direction.name || direction.id} 围绕主题形成了一组可比较的研究问题、方法和结论。`,
    directions: [direction.id],
    paperCount: direction.paper_count || (direction.papers || []).length,
    papers: representativePaperIds(direction, 3)
  }));
}

function renderDomainInsights(corpus) {
  const rows = domainInsightRows(corpus);
  if (!rows.length) {
    return `<p class="summary-text">当前样本不足以形成稳定的领域综合洞察。建议补充更多方向和代表文献后再生成宏观判断。</p>`;
  }
  return `
    <div class="domain-insight-grid">
      ${rows.map((item) => `
        <article class="domain-insight-card">
          <div>
            <h4>${escapeHtml(item.title)}</h4>
            <p>${escapeHtml(compactText(item.summary, 220))}</p>
          </div>
          <div class="insight-meta">
            ${item.directions.length ? `<span>${item.directions.map((id) => escapeHtml(id)).join(" / ")}</span>` : ""}
            <span>${escapeHtml(item.paperCount || 0)} 篇支撑文献</span>
          </div>
          ${item.papers.length ? `<div class="insight-links">${evidencePaperLinks(item.papers)}</div>` : ""}
        </article>
      `).join("")}
    </div>
  `;
}

function consensusItems(value, fallbackValue) {
  const source = asArray(value).length ? asArray(value) : pointItems(fallbackValue, 4, 180);
  return source.map((item) => cleanText(item)).filter(Boolean).slice(0, 4);
}

function renderConsensusMatrix(corpus) {
  const consensus = consensusItems(corpus.research_consensus, corpus.commonality);
  const divergence = consensusItems(corpus.research_disagreements || corpus.research_divergence, corpus.differences);
  const opportunities = consensusItems(corpus.future_opportunities, corpus.gap);
  const columns = [
    ["研究共识", consensus],
    ["研究分歧", divergence],
    ["未来机会", opportunities]
  ];
  return `
    <div class="consensus-grid">
      ${columns.map(([title, items]) => `
        <article class="consensus-card">
          <h4>${escapeHtml(title)}</h4>
          <ul>
            ${(items.length ? items : ["当前证据不足，暂不形成稳定判断。"]).map((item) => `<li>${inlineMarkdown(compactText(item, 180))}</li>`).join("")}
          </ul>
        </article>
      `).join("")}
    </div>
  `;
}

function renderLatexFormula(value) {
  const source = String(value ?? "").trim();
  if (!source) return "";
  const normalized = source
    .replace(/\r\n/g, "\n")
    .replace(/\\\]\s*\\\[/g, "\\]\n\\[")
    .replace(/\$\$\s*\$\$/g, "$$\n$$");
  const displayMatches = normalized.match(/\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g) || [];
  if (displayMatches.length > 1) {
    return displayMatches
      .map((item) => `<div class="formula-display-line">${escapeHtml(wrapDisplayLatex(item))}</div>`)
      .join("");
  }
  const formulaSegments = splitDisplayFormulaSegments(normalized);
  if (formulaSegments.length > 1) {
    return formulaSegments
      .map((item) => `<div class="formula-display-line">${escapeHtml(wrapDisplayLatex(item))}</div>`)
      .join("");
  }
  return escapeHtml(wrapDisplayLatex(normalized));
}

function stripLatexDelimiters(value) {
  let source = String(value ?? "").trim();
  if (source.startsWith("\\[") && source.endsWith("\\]")) return source.slice(2, -2).trim();
  if (source.startsWith("\\(") && source.endsWith("\\)")) return source.slice(2, -2).trim();
  if (source.startsWith("$$") && source.endsWith("$$")) return source.slice(2, -2).trim();
  if (/^\$[^$][\s\S]*[^$]\$$/.test(source)) return source.slice(1, -1).trim();
  return source;
}

function maskLatexEnvironments(source) {
  return String(source ?? "").replace(
    /\\begin\{(aligned|alignedat|cases|array|matrix|bmatrix|pmatrix|vmatrix|Vmatrix|smallmatrix)\}[\s\S]*?\\end\{\1\}/g,
    (match) => " ".repeat(match.length)
  );
}

function formulaSegmentHasRelation(value) {
  return /(=|\\leq?|\\geq?|\\in\b|\\approx\b|\\sim\b|\\arg\s*\\?(?:min|max)|\\(?:min|max)\b|s\.?\s*t\.?|\\text\{s\.?\s*t\.?\})/.test(String(value ?? ""));
}

function cleanFormulaSegment(value) {
  return String(value ?? "")
    .replace(/^[\s;；,，]*(?:\\quad|\\qquad|\\;|\\,|\s)+/g, "")
    .replace(/[\s;；,，]+$/g, "")
    .trim();
}

function splitDisplayFormulaSegments(value) {
  const body = stripLatexDelimiters(value);
  if (!body) return [];
  if (/\\begin\{(aligned|alignedat|cases|array|matrix|bmatrix|pmatrix|vmatrix|Vmatrix|smallmatrix)\}/.test(body)) {
    return [body];
  }
  const parts = [];
  let depth = 0;
  let start = 0;
  const pushPart = (end, nextStart) => {
    const previous = cleanFormulaSegment(body.slice(start, end));
    const next = cleanFormulaSegment(body.slice(nextStart));
    if (previous && next && formulaSegmentHasRelation(previous) && formulaSegmentHasRelation(next)) {
      parts.push(previous);
      start = nextStart;
      return true;
    }
    return false;
  };

  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    const escaped = body[index - 1] === "\\";
    if (char === "{" && !escaped) depth += 1;
    if (char === "}" && !escaped) depth = Math.max(0, depth - 1);
    if (depth !== 0) continue;
    if (body.startsWith("\\\\", index)) {
      if (pushPart(index, index + 2)) index += 1;
      continue;
    }
    if (char === ";" || char === "；") {
      pushPart(index, index + 1);
    }
  }

  const tail = cleanFormulaSegment(body.slice(start));
  if (tail) parts.push(tail);
  return parts.length > 1 ? parts : [body];
}

function sanitizeDisplayLatexBody(value) {
  let body = normalizeLatexSource(stripLatexDelimiters(value))
    .replace(/[，,]\s*\\\\/g, " \\\\")
    .trim();
  if (!body) return "";

  const alignedBegin = (body.match(/\\begin\{aligned\}/g) || []).length;
  const alignedEnd = (body.match(/\\end\{aligned\}/g) || []).length;
  if (alignedBegin > alignedEnd) {
    body += " " + "\\end{aligned}".repeat(alignedBegin - alignedEnd);
  } else if (alignedEnd > alignedBegin) {
    let extra = alignedEnd - alignedBegin;
    body = body.replace(/\\end\{aligned\}/g, (match) => {
      if (extra > 0) {
        extra -= 1;
        return "";
      }
      return match;
    }).trim();
  }

  const masked = maskLatexEnvironments(body);
  const hasTopLevelAlignment = masked.includes("&");
  const hasTopLevelRows = /(^|[^\\])\\\\(?![A-Za-z])/.test(masked);
  const hasAligned = /\\begin\{aligned\}/.test(body);
  if ((hasTopLevelAlignment || hasTopLevelRows) && !hasAligned) {
    body = `\\begin{aligned} ${body} \\end{aligned}`;
  }
  return body;
}

function wrapDisplayLatex(value) {
  const body = sanitizeDisplayLatexBody(value);
  return body ? `\\[${body}\\]` : "";
}

function renderInlineLatex(value) {
  let source = String(value ?? "").trim();
  if (!source) return "";
  const parsed = splitVariableSymbol(source);
  source = parsed.symbol || source;
  source = source
    .replace(/^\\\(/, "")
    .replace(/\\\)$/, "")
    .replace(/^\\\[/, "")
    .replace(/\\\]$/, "")
    .replace(/^\$\$/, "")
    .replace(/\$\$$/, "")
    .replace(/^\$/, "")
    .replace(/\$$/, "")
    .trim();
  source = normalizeInlineLatexSource(source);
  return `\\(${escapeHtml(source)}\\)`;
}

function splitVariableSymbol(value) {
  let source = String(value ?? "").trim();
  if (!source) return {symbol: "", tail: ""};
  source = source
    .replace(/^\\\(/, "")
    .replace(/\\\)$/, "")
    .replace(/^\\\[/, "")
    .replace(/\\\]$/, "")
    .replace(/^\$\$/, "")
    .replace(/\$\$$/, "")
    .replace(/^\$/, "")
    .replace(/\$$/, "")
    .trim();

  const accent = splitAccentLatexSymbol(source);
  if (accent) return accent;

  const cjkMatch = source.match(/^([^\u4e00-\u9fff]+)([\u4e00-\u9fff].*)$/);
  if (cjkMatch) {
    let symbol = cjkMatch[1].trim();
    let tail = cjkMatch[2].trim();
    const lastBrace = symbol.lastIndexOf("}");
    if (lastBrace >= 0 && lastBrace < symbol.length - 1) {
      const suffix = symbol.slice(lastBrace + 1).trim();
      if (/^[A-Za-z][A-Za-z0-9-]*$/.test(suffix)) {
        symbol = symbol.slice(0, lastBrace + 1).trim();
        tail = `${suffix}${tail}`;
      }
    }
    return {symbol, tail};
  }

  if (/[\\_^{}]/.test(source)) {
    return {symbol: source, tail: ""};
  }

  const commandMatch = source.match(/^(\\[A-Za-z]+(?:\s+[A-Za-z])?(?:(?:_\{[^{}]+\}|_[A-Za-z0-9Α-Ωα-ω]+|\^\{[^{}]+\}|\^[A-Za-z0-9Α-Ωα-ω]+)*))/);
  const latinMatch = source.match(/^([A-Za-zΑ-Ωα-ω]+(?:(?:_\{[^{}]+\}|_[A-Za-z0-9Α-Ωα-ω]+|\^\{[^{}]+\}|\^[A-Za-z0-9Α-Ωα-ω]+)*))/);
  const match = commandMatch || latinMatch;
  if (!match) return {symbol: source, tail: ""};
  const symbol = match[1].trim();
  const tail = source.slice(match[0].length).trim();
  return {symbol, tail};
}

function readLatexGroup(source, start) {
  if (source[start] !== "{") return null;
  let depth = 0;
  for (let index = start; index < source.length; index += 1) {
    const char = source[index];
    if (char === "{" && source[index - 1] !== "\\") depth += 1;
    if (char === "}" && source[index - 1] !== "\\") {
      depth -= 1;
      if (depth === 0) {
        return {text: source.slice(start, index + 1), end: index + 1};
      }
    }
  }
  return null;
}

function readLatexAtom(source, start) {
  let pos = start;
  while (/\s/.test(source[pos] || "")) pos += 1;
  if (!source[pos]) return null;
  if (source[pos] === "{") return readLatexGroup(source, pos);
  if (source[pos] === "\\") {
    const command = source.slice(pos).match(/^\\[A-Za-z]+/);
    if (command) return {text: command[0], end: pos + command[0].length};
  }
  return {text: source[pos], end: pos + 1};
}

function readLatexScripts(source, start) {
  let pos = start;
  let text = "";
  while (pos < source.length) {
    const before = pos;
    while (/\s/.test(source[pos] || "")) pos += 1;
    const marker = source[pos];
    if (marker !== "_" && marker !== "^") {
      pos = before;
      break;
    }
    pos += 1;
    const atom = readLatexAtom(source, pos);
    if (!atom) {
      pos = before;
      break;
    }
    text += source.slice(before, atom.end);
    pos = atom.end;
  }
  return {text, end: pos};
}

function splitAccentLatexSymbol(source) {
  const command = String(source ?? "").match(/^\\(tilde|hat|bar|overline|underline|vec|dot|ddot|widehat|widetilde)\b/);
  if (!command) return null;
  const atom = readLatexAtom(source, command[0].length);
  if (!atom) return null;
  const scripts = readLatexScripts(source, atom.end);
  return {
    symbol: `${command[0]}${atom.text}${scripts.text}`,
    tail: source.slice(scripts.end).trim()
  };
}

function typesetMath() {
  const run = (attempt = 0) => {
    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([canvas]).catch((error) => console.warn("MathJax typeset failed", error));
    } else if (attempt < 20) {
      setTimeout(() => run(attempt + 1), 250);
    }
  };
  if (window.MathJax?.startup?.promise) {
    window.MathJax.startup.promise.then(run);
  } else {
    setTimeout(() => run(0), 250);
  }
}

function metric(label, value) {
  return `<article class="metric"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></article>`;
}

function triad(commonality, differences, gap) {
  return `
    <div class="triad-grid">
      <article class="triad-card"><h4>共性问题</h4>${pointList(commonality, 4, 180)}</article>
      <article class="triad-card"><h4>差异问题</h4>${pointList(differences, 4, 180)}</article>
      <article class="triad-card"><h4>研究 Gap</h4>${pointList(gap, 4, 180)}</article>
    </div>
  `;
}

function setHeader(layer, title, subtitle) {
  const layerText = {
    overview: "第 1 层 · 总主题层",
    direction: "第 2 层 · 研究方向层",
    paper: "第 3 层 · 单篇文献层",
    reproduction: "论文复现 · 工具链展示"
  };
  layerEyebrow.textContent = layerText[layer] || "";
  pageTitle.textContent = title;
  pageSubtitle.textContent = subtitle;
  backButton.href = layer === "paper" && currentDirection ? directionUrl(currentDirection.id) : overviewUrl();
  backButton.classList.toggle("hidden", layer === "reproduction");
  overviewButton.textContent = layer === "reproduction" ? "返回综述" : "总览";
  renderBreadcrumb(layer);
}

function renderBreadcrumb(layer) {
  if (layer === "reproduction") {
    breadcrumb.innerHTML = `<a href="${overviewUrl()}">文献综述</a><span>/</span><span>论文复现</span>`;
    return;
  }
  const items = [`<a href="${overviewUrl()}">总览</a>`];
  if (currentDirection) {
    items.push(`<span>/</span><a href="${directionUrl(currentDirection.id)}">${escapeHtml(currentDirection.name)}</a>`);
  }
  if (layer === "paper" && currentPaper) {
    items.push(`<span>/</span><span>${escapeHtml(currentPaper.title_cn)}</span>`);
  }
  breadcrumb.innerHTML = items.join("");
}

function renderOverview() {
  currentDirection = null;
  currentPaper = null;
  const corpus = reviewData.corpus || {};
  setHeader("overview", reviewData.topic || "文献综述总览", "从整体数据、方向地图、研究脉络和 Gap 开始阅读。");
  canvas.innerHTML = `
    <section class="metrics">
      ${metric("文献总数", `${corpus.paper_total || 0} 篇`)}
      ${metric("时间范围", corpus.year_range || "-")}
      ${metric("研究方向", `${corpus.direction_total || (reviewData.directions || []).length} 个`)}
    </section>

    <section class="band overview-band">
      <div class="band-title overview-band-title">
        <h3>领域概览</h3>
      </div>
      ${renderCorpusOverview(corpus)}
    </section>

    <section class="band">
      <div class="band-title">
        <h3>研究方向地图</h3>
        <small>点击卡片打开第 2 层新页面</small>
      </div>
      <div class="direction-grid">
        ${(reviewData.directions || []).map((direction) => `
          <article class="direction-card">
            <div>
              <h4>${escapeHtml(direction.id)} · ${escapeHtml(direction.name)}</h4>
              <small>${escapeHtml(direction.name_en)}</small>
            </div>
            <p>${escapeHtml(direction.summary)}</p>
            <div class="card-meta">
              <span>${escapeHtml(direction.paper_count)} 篇文献</span>
              <span>热度 ${escapeHtml(direction.heat)}</span>
            </div>
            ${tags(direction.keywords)}
            <a class="card-button" href="${directionUrl(direction.id)}">进入方向分析</a>
          </article>
        `).join("")}
      </div>
    </section>

    <section class="band">
      <div class="band-title"><h3>研究方向对比</h3></div>
      ${triad(corpus.commonality, corpus.differences, corpus.gap)}
    </section>

    <section class="band">
      <div class="band-title">
        <h3>研究脉络</h3>
        <small>按方向梳理问题、方法和代表性趋势</small>
      </div>
      ${renderInsightTimeline(corpus)}
    </section>

    <section class="band">
      <div class="band-title">
        <h3>领域综合洞察</h3>
        <small>从全部方向和代表文献中提炼宏观判断</small>
      </div>
      ${renderDomainInsights(corpus)}
      ${renderConsensusMatrix(corpus)}
    </section>
  `;
  typesetMath();
}

function renderBars(direction) {
  const max = Math.max(...(direction.methods_distribution || []).map((item) => item.occurrence_count || item.count || 0), 1);
  return `
    <div class="mini-bars">
      ${(direction.methods_distribution || []).map((item) => `
        <div class="bar-row">
          <span>${escapeHtml(item.method)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.round((item.occurrence_count || item.count || 0) / max * 100)}%"></div></div>
          <b title="覆盖论文数 / 出现次数">${escapeHtml(item.paper_count || 0)}/${escapeHtml(item.occurrence_count || item.count || 0)}</b>
        </div>
      `).join("")}
      <small class="bar-note">数字为“覆盖论文数 / 出现次数”。一篇论文对同一方法只计 1 次覆盖。</small>
    </div>
  `;
}

function renderDirectionMethodSummary(direction) {
  const methods = (direction.methods_distribution || [])
    .map((item) => cleanText(item.method))
    .filter(Boolean)
    .slice(0, 5);
  if (!methods.length) return pointList(direction.summary, 2, 120);
  const families = [];
  if (methods.some((item) => /优化|规划|MILP|随机|调度|配置|allocation|optimization/i.test(item))) {
    families.push(`优化建模：${methods.filter((item) => /优化|规划|MILP|随机|调度|配置|allocation|optimization/i.test(item)).slice(0, 3).join("、")}`);
  }
  if (methods.some((item) => /学习|强化|预测|数据|learning|forecast/i.test(item))) {
    families.push(`数据驱动：${methods.filter((item) => /学习|强化|预测|数据|learning|forecast/i.test(item)).slice(0, 3).join("、")}`);
  }
  if (methods.some((item) => /博弈|均衡|市场|竞价|auction|game|bidding/i.test(item))) {
    families.push(`市场机制：${methods.filter((item) => /博弈|均衡|市场|竞价|auction|game|bidding/i.test(item)).slice(0, 3).join("、")}`);
  }
  if (methods.some((item) => /综述|评估|比较|收益|valuation|assessment|review/i.test(item))) {
    families.push(`评估归纳：${methods.filter((item) => /综述|评估|比较|收益|valuation|assessment|review/i.test(item)).slice(0, 3).join("、")}`);
  }
  const items = families.length ? families : [`主要采用${methods.slice(0, 4).join("、")}等方法。`];
  return `<ul class="point-list method-summary-list">${items.slice(0, 3).map((item) => `<li>${escapeHtml(compactText(item, 90))}</li>`).join("")}</ul>`;
}

function paperSearchText(paper) {
  return [
    paper.title,
    paper.title_cn,
    paper.authors,
    paper.year,
    paper.research_problem,
    paper.method,
    paper.scenario,
    paper.conclusion,
    paper.innovation,
    paper.limitation,
    ...(paper.keywords || []),
  ].join(" ").toLowerCase();
}

function paperMethodText(paper) {
  return [paper.method, paper.innovation, ...(paper.keywords || [])].join(" ").toLowerCase();
}

function methodOptions(direction) {
  return [...new Set((direction.methods_distribution || []).map((item) => item.method).filter(Boolean))];
}

function renderPaperRows(direction, papers) {
  if (!papers.length) {
    return `<tr><td colspan="7" class="empty-cell">没有符合筛选条件的文献。</td></tr>`;
  }
  return papers.map((paper) => `
    <tr>
      <td><a class="paper-link" href="${paperUrl(direction.id, paper.id)}">${escapeHtml(compactText(paper.title_cn, 160))}</a><br><small>${escapeHtml(compactText(paper.authors, 80))} · ${escapeHtml(paper.year)}</small></td>
      <td>${pointList(paper.research_problem, 2, 200)}</td>
      <td>${pointList(paper.method, 2, 200)}</td>
      <td>${pointList(paper.scenario, 2, 200)}</td>
      <td>${pointList(paper.conclusion, 2, 200)}</td>
      <td>${pointList(paper.innovation, 2, 200)}</td>
      <td>${pointList(paper.limitation, 2, 200)}</td>
    </tr>
  `).join("");
}

function applyDirectionFilters() {
  if (!currentDirection) return;
  const query = document.querySelector("#paperFilterQuery")?.value.trim().toLowerCase() || "";
  const method = document.querySelector("#paperFilterMethod")?.value || "";
  const year = document.querySelector("#paperFilterYear")?.value || "";
  const papers = (currentDirection.papers || []).filter((paper) => {
    const matchesQuery = !query || paperSearchText(paper).includes(query);
    const matchesMethod = !method || paperMethodText(paper).includes(method.toLowerCase());
    const matchesYear = !year || String(paper.year) === year;
    return matchesQuery && matchesMethod && matchesYear;
  });
  const body = document.querySelector("#directionPaperRows");
  const count = document.querySelector("#paperFilterCount");
  if (body) body.innerHTML = renderPaperRows(currentDirection, papers);
  if (count) count.textContent = `${papers.length} / ${(currentDirection.papers || []).length} 篇`;
}

function bindDirectionFilters() {
  ["#paperFilterQuery", "#paperFilterMethod", "#paperFilterYear"].forEach((selector) => {
    const el = document.querySelector(selector);
    if (el) el.addEventListener("input", applyDirectionFilters);
  });
  const reset = document.querySelector("#paperFilterReset");
  if (reset) {
    reset.addEventListener("click", () => {
      ["#paperFilterQuery", "#paperFilterMethod", "#paperFilterYear"].forEach((selector) => {
        const el = document.querySelector(selector);
        if (el) el.value = "";
      });
      applyDirectionFilters();
    });
  }
}

function renderDirectionComparison(activeDirection) {
  return `
    <section class="band">
      <div class="band-title">
        <h3>方向对比表</h3>
        <small>比较各方向的问题、方法、共性和空白</small>
      </div>
      <div class="paper-table comparison-table">
        <table>
          <thead>
            <tr>
              <th>方向</th>
              <th>文献数</th>
              <th>核心问题</th>
              <th>主要方法</th>
              <th>方向共性</th>
              <th>研究空白</th>
            </tr>
          </thead>
          <tbody>
            ${(reviewData.directions || []).map((direction) => `
              <tr class="${direction.id === activeDirection.id ? "active-row" : ""}">
                <td><a class="paper-link" href="${directionUrl(direction.id)}">${escapeHtml(direction.id)} · ${escapeHtml(direction.name)}</a></td>
                <td>${escapeHtml(direction.paper_count)} 篇</td>
                <td>${pointList(direction.core_question, 2, 200)}</td>
                <td>${renderDirectionMethodSummary(direction)}</td>
                <td>${pointList(direction.commonality, 2, 200)}</td>
                <td>${pointList(direction.gap, 2, 200)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function renderKnowledgeStrip(direction) {
  return `
    <section class="band knowledge-band">
      <div class="band-title">
        <h3>方向知识卡片</h3>
        <small>核心问题、关键词、方法分布与趋势</small>
      </div>
      <div class="knowledge-strip">
        <article>
          <small>核心问题</small>
          <p>${escapeHtml(compactText(direction.core_question, 130))}</p>
        </article>
        <article>
          <small>高频关键词</small>
          ${tags(direction.keywords)}
        </article>
        <article class="method-panel">
          <small>主要方法分布</small>
          ${renderBars(direction)}
        </article>
        <article>
          <small>发展趋势</small>
          <p>${escapeHtml(compactText(direction.knowledge_card?.trend, 130))}</p>
        </article>
      </div>
    </section>
  `;
}

function renderDirection(directionId) {
  currentDirection = getDirection(directionId);
  currentPaper = null;
  if (!currentDirection) {
    renderOverview();
    return;
  }
  setHeader("direction", currentDirection.name, "比较同一方向下文献的研究问题、方法、场景、结论和局限。");
  canvas.innerHTML = `
    <section class="band">
      <div class="band-title">
        <div>
          <h3>${escapeHtml(currentDirection.id)} · ${escapeHtml(currentDirection.name)}</h3>
          <small>${escapeHtml(currentDirection.name_en)}</small>
        </div>
        <small>${escapeHtml(currentDirection.paper_count)} 篇文献 · 热度 ${escapeHtml(currentDirection.heat)}</small>
      </div>
      <p class="summary-text">${escapeHtml(cleanText(currentDirection.summary))}</p>
    </section>

    ${renderDirectionComparison(currentDirection)}

    ${renderKnowledgeStrip(currentDirection)}

    <section class="band">
      <div class="band-title">
        <h3>方向内文献对比</h3>
        <small id="paperFilterCount">${escapeHtml((currentDirection.papers || []).length)} / ${escapeHtml((currentDirection.papers || []).length)} 篇</small>
      </div>
      <div class="filter-bar">
        <label>
          <span>搜索</span>
          <input id="paperFilterQuery" type="search" placeholder="标题、方法、结论、关键词">
        </label>
        <label>
          <span>方法</span>
          <select id="paperFilterMethod">
            <option value="">全部方法</option>
            ${methodOptions(currentDirection).map((method) => `<option value="${escapeHtml(method)}">${escapeHtml(method)}</option>`).join("")}
          </select>
        </label>
        <label>
          <span>年份</span>
          <select id="paperFilterYear">
            <option value="">全部年份</option>
            ${[...new Set((currentDirection.papers || []).map((paper) => paper.year).filter(Boolean))].sort().map((year) => `<option value="${escapeHtml(year)}">${escapeHtml(year)}</option>`).join("")}
          </select>
        </label>
        <button id="paperFilterReset" type="button">重置</button>
      </div>
      <div class="paper-table">
        <table>
          <thead>
            <tr>
              <th>文献</th>
              <th>研究问题</th>
              <th>研究方法</th>
              <th>数据/场景</th>
              <th>主要结论</th>
              <th>创新点</th>
              <th>局限/GAP</th>
            </tr>
          </thead>
          <tbody id="directionPaperRows">
            ${renderPaperRows(currentDirection, currentDirection.papers || [])}
          </tbody>
        </table>
      </div>
    </section>

    <section class="band">
      <div class="band-title"><h3>方向总结</h3></div>
      ${triad(currentDirection.commonality, currentDirection.differences, currentDirection.gap)}
    </section>
  `;
  bindDirectionFilters();
  typesetMath();
}

function paperDetail(paper, direction) {
  const webReview = paper.web_review || {};
  const brief = webReview.brief_sections || {};
  const detailed = webReview.detailed_sections || {};
  const inputText = (paper.method_inputs || []).join("、") || "价格场景、资源容量、初始状态 \\(SOC_0\\)、市场规则、风险偏好参数 \\(\\lambda\\)";
  const outputText = (paper.method_outputs || []).join("、") || "最优报价曲线、收益分解、风险指标和关键约束影子价格";
  const parameterText = (paper.method_parameters || []).join("、") || "功率边界、SOC 动态、能量守恒、市场申报边界、服务履约要求";
  const methodText = paper.method_detail || `**方法主线：**${paper.method}

该方法可拆成四类对象：

- **输入参数**：${inputText}。
- **决策变量**：结合论文对象定义，例如充放电功率 \\(p_t^{ch}\\)、放电功率 \\(p_t^{dis}\\)、市场申报量 \\(q_t\\) 或服务容量 \\(r_t\\)。
- **核心约束/参数**：${parameterText}。
- **输出结果**：${outputText}。`;
  const fallbackFormulaItems = [
    {
      name: "收益最大化目标",
      formula: "\\[\\max_{p_t^{ch},p_t^{dis},r_t}\\; \\sum_{t\\in T}\\left(\\pi_t^e p_t^{dis}-\\pi_t^e p_t^{ch}+\\pi_t^s r_t\\right)-C_{deg}(p_t^{ch},p_t^{dis})-\\lambda\\,Risk\\]",
      note: "其中 \\(\\pi_t^e\\) 表示能量价格，\\(\\pi_t^s\\) 表示服务价格，\\(r_t\\) 表示辅助服务或容量申报量，\\(C_{deg}\\) 表示电池退化成本。"
    },
    {
      name: "SOC 动态约束",
      formula: "\\[SOC_t=SOC_{t-1}+\\eta_{ch}p_t^{ch}\\Delta t-\\frac{p_t^{dis}\\Delta t}{\\eta_{dis}},\\quad SOC^{min}\\le SOC_t\\le SOC^{max}\\]",
      note: "该式保证储能状态随充放电行为连续变化，并受 \\(SOC^{min}\\) 与 \\(SOC^{max}\\) 边界限制。"
    },
    {
      name: "风险或不确定性项",
      formula: "\\[Risk=CVaR_{\\alpha}(Loss)\\quad \\text{or}\\quad \\max_{\\xi\\in\\Omega} Cost(x,\\xi)\\]",
      note: "随机优化常用 \\(CVaR_{\\alpha}\\) 控制尾部亏损，鲁棒优化则在不确定性集合 \\(\\Omega\\) 内寻找最坏场景。"
    }
  ];
  const formulaItems = (paper.formulas && paper.formulas.length) ? paper.formulas : fallbackFormulaItems;
  const sections = {
    coreSummary: cleanText(webReview.core_summary_cn || `${paper.research_problem} 文章采用${paper.method}，在${paper.scenario}场景下分析问题，并得到${paper.conclusion}。其主要贡献是${paper.innovation}，局限与后续改进方向包括${paper.limitation}。`),
    problemBrief: cleanText(brief.research_problem_gap_cn || `${paper.research_problem} 该问题的 gap 主要体现在 ${paper.background || paper.limitation}`),
    methodBrief: cleanText(brief.method_innovation_cn || `${paper.method} 创新点在于 ${paper.innovation}`),
    conclusionBrief: cleanText(brief.main_conclusion_cn || paper.conclusion),
    limitationBrief: cleanText(brief.limitation_future_cn || `${paper.limitation} 后续可从真实市场数据、跨市场联合报价和模型可解释性方面改进。`),
    problemDetail: cleanText(detailed.research_problem_detail_cn || `${paper.background} 具体而言，论文关注 ${paper.research_problem}。这一问题的难点在于市场价格、储能物理约束和风险偏好之间存在耦合，已有研究往往难以同时处理收益、约束可行性和泛化能力。`),
    methodOverview: cleanText(detailed.method_overview_cn || `${paper.method} 该方法用于把价格、资源状态和市场规则转化为可求解的报价或调度决策，并通过约束、风险项或学习策略平衡收益和运行安全。`),
    conclusionDetail: cleanText(detailed.main_conclusion_detail_cn || `${paper.conclusion} 该结论说明方法不仅服务于收益提升，也帮助识别市场规则、储能状态和不确定性之间的关键权衡。`),
    limitationDetail: cleanText(detailed.limitation_future_detail_cn || `${paper.limitation} 未来可以补充真实市场结算数据、考虑更多市场联动、校准储能退化参数，并检验策略在不同电价和资源条件下的稳健性。`)
  };
  const methodSteps = buildMethodSteps(paper, webReview.method_steps, formulaItems, {inputText, outputText, parameterText});
  return {methodText, formulaItems, direction, sections, methodSteps};
}

function conciseStepName(value, index) {
  const fallback = ["问题建模", "变量定义", "策略求解", "结果评估", "稳健检验"][index] || `步骤 ${index + 1}`;
  const text = cleanText(value).replace(/^[-*\d.、\s]+/, "");
  if (!text) return fallback;
  const phrase = text.split(/[，,；;。:：]/)[0].trim();
  return phrase.length <= 14 ? phrase : phrase.slice(0, 14);
}

function formulaSearchText(formula) {
  const variables = (formula.variables || [])
    .map((item) => `${item.symbol || ""} ${item.meaning || item.meaning_cn || ""} ${item.unit || ""}`)
    .join(" ");
  return [
    formula.id,
    formula.original_number,
    formula.name,
    formula.formula,
    formula.note,
    formula.used_for,
    variables
  ].join(" ").toLowerCase();
}

function formulaMatchesRef(formula, ref) {
  const target = String(ref || "").trim().toLowerCase();
  if (!target) return false;
  return [formula.id, formula.original_number, formula.name]
    .map((item) => String(item || "").trim().toLowerCase())
    .some((item) => item && item === target);
}

function formulaSemanticScore(step, formula) {
  const stepText = `${step.name || ""} ${step.detail || ""}`.toLowerCase();
  const formulaText = formulaSearchText(formula);
  const keywordGroups = [
    ["soc", "荷电", "state of charge", "状态"],
    ["收益", "收入", "利润", "reward", "目标", "objective", "net"],
    ["折旧", "退化", "degradation", "成本", "cost"],
    ["动作", "充电", "放电", "功率", "action", "power", "阈值"],
    ["单调", "monotonic", "导数", "derivative"],
    ["离散", "聚类", "kmeans", "reference", "参考水平"],
    ["风险", "cvar", "robust", "uncertainty", "不确定"],
    ["约束", "constraint", "边界", "可行"],
    ["市场", "出清", "价格", "clearing", "price", "bidding", "报价"]
  ];
  let score = 0;
  for (const group of keywordGroups) {
    const inStep = group.some((word) => stepText.includes(word));
    const inFormula = group.some((word) => formulaText.includes(word));
    if (inStep && inFormula) score += 3;
  }
  const formulaWords = formulaText
    .split(/[\s,，;；:：。()（）{}[\]_^+\-=\\/|]+/)
    .filter((word) => word.length >= 2 && !/^\d+$/.test(word));
  for (const word of new Set(formulaWords)) {
    if (stepText.includes(word)) score += 1;
  }
  return score;
}

function resolveStepFormulas(step, formulaItems, allowSemanticFallback) {
  const refs = asArray(step.formulaRefs).map((item) => String(item || "").trim()).filter(Boolean);
  const direct = refs
    .flatMap((ref) => formulaItems.filter((formula) => formulaMatchesRef(formula, ref)))
    .filter((formula, index, array) => array.findIndex((item) => item.id === formula.id && item.formula === formula.formula) === index);
  if (direct.length) return direct.slice(0, 2);
  if (refs.length || !allowSemanticFallback) return [];
  const scored = formulaItems
    .map((formula) => ({formula, score: formulaSemanticScore(step, formula)}))
    .filter((item) => item.score >= 3)
    .sort((a, b) => b.score - a.score);
  return scored.slice(0, 1).map((item) => item.formula);
}

function renderFormulaVariables(formula) {
  const variables = (formula.variables || []).filter((item) => item.symbol || item.meaning || item.meaning_cn);
  if (!variables.length) return "";
  return `
    <ul class="formula-vars">
      ${variables.slice(0, 6).map((item) => {
        const parsed = splitVariableSymbol(item.symbol || "");
        const meaning = cleanText([parsed.tail, item.meaning || item.meaning_cn || ""].filter(Boolean).join(" "));
        return `
          <li>
            ${parsed.symbol ? `<b class="var-symbol">${renderInlineLatex(parsed.symbol)}</b>` : ""}
            <span class="var-meaning">${inlineMarkdown(meaning)}</span>
            ${item.unit && item.unit !== "unknown" ? `<span class="var-unit">${escapeHtml(item.unit)}</span>` : ""}
          </li>
        `;
      }).join("")}
    </ul>
  `;
}

function buildMethodSteps(paper, promptSteps, formulaItems, fallbackTexts) {
  const hasPromptSteps = asArray(promptSteps).length > 0;
  const rawSteps = hasPromptSteps
    ? asArray(promptSteps).map((item) => ({
        name: item.step_name || item.name || item.title,
        detail: item.step_detail_cn || item.detail_cn || item.explanation_cn,
        formulaRefs: asArray(item.formula_refs)
      }))
    : asArray(paper.method_flow).map((step) => ({name: step, detail: ""}));
  const baseSteps = rawSteps.length ? rawSteps : [
    {name: "场景建模", detail: `整理市场价格、资源容量和运行状态等输入：${fallbackTexts.inputText}。`},
    {name: "变量定义", detail: "定义充放电功率、市场申报量、服务容量和状态变量，使报价问题可以被模型表达。"},
    {name: "约束求解", detail: `加入核心约束和参数：${fallbackTexts.parameterText}，并求解满足物理和市场规则的策略。`},
    {name: "结果评估", detail: `输出 ${fallbackTexts.outputText}，比较收益、风险、约束违约和市场表现。`}
  ];
  return baseSteps.slice(0, 10).map((step, index) => {
    const normalizedStep = {
      name: conciseStepName(step.name, index),
      detail: compactText(step.detail || `${step.name} 环节用于支撑 ${paper.method}，帮助从输入信息推导报价、调度或市场响应结果。`, 300),
      formulaRefs: step.formulaRefs
    };
    return {
      ...normalizedStep,
      formulas: resolveStepFormulas(normalizedStep, formulaItems, !hasPromptSteps)
    };
  });
}

function renderBriefSections(detail) {
  const cards = [
    ["研究问题", "突出 gap", detail.sections.problemBrief],
    ["研究方法", "突出创新点", detail.sections.methodBrief],
    ["主要结论", "突出发现", detail.sections.conclusionBrief],
    ["局限性", "指出改进方向", detail.sections.limitationBrief]
  ];
  return `
    <div class="module-grid compact brief-grid">
      ${cards.map(([title, subtitle, text]) => `
        <article class="module">
          <div class="module-head">
            <h4>${escapeHtml(title)}</h4>
            <small>${escapeHtml(subtitle)}</small>
          </div>
          ${sectionPointList(text, 3, 300)}
        </article>
      `).join("")}
    </div>
  `;
}

function paperDisplayTitle(paper) {
  return paper?.title_cn || paper?.title || paper?.id || "";
}

function renderRelatedPapers(paper) {
  const related = (paper.related_papers || [])
    .map((paperRef) => {
      const ref = String(paperRef || "").trim();
      if (!ref) return null;
      const location = findPaperLocation(ref);
      if (!location) return null;
      return {
        title: paperDisplayTitle(location.paper),
        meta: [location.direction.id, location.paper.year].filter(Boolean).join(" · "),
        url: paperUrl(location.direction.id, location.paper.id)
      };
    })
    .filter(Boolean);
  if (!related.length) return "";
  return `
    <div class="related-paper-list">
      ${related.map((item) => `
        <a class="related-paper-item" href="${escapeHtml(item.url)}">
          <span>${inlineMarkdown(item.title)}</span>
          ${item.meta ? `<small>${escapeHtml(item.meta)}</small>` : ""}
        </a>
      `).join("")}
    </div>
  `;
}

function renderKeywordRelatedSection(paper) {
  const relatedHtml = renderRelatedPapers(paper);
  const keywordHtml = tags(paper.keywords);
  if (!relatedHtml && !(paper.keywords || []).length) return "";
  return `
    <section class="band keyword-related-band">
      <div class="band-title"><h3>关键词与相似文献</h3></div>
      ${keywordHtml}
      ${relatedHtml ? `<div class="related-block"><h4>相似文献</h4>${relatedHtml}</div>` : ""}
    </section>
  `;
}

function runAssetUrl(relativePath) {
  const path = String(relativePath || "").replace(/\\/g, "/").replace(/^\/+/, "");
  if (!path || !activeRunId) return "";
  return `/runs/${encodeURIComponent(activeRunId)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function visualAssetLabel(asset, index = 0) {
  const isTable = String(asset.kind || "").toLowerCase() === "table";
  const rawId = String(asset.id || "").trim();
  const idNumber = (rawId.match(/\d+/) || [])[0] || "";
  const captionNumber = (String(asset.caption || "").match(/(?:fig(?:ure)?|tab(?:le)?)\s*\.?\s*(\d+)/i) || [])[1] || "";
  const number = idNumber || captionNumber;
  const numberText = number ? ` ${number}` : "";
  const kind = isTable ? `表${numberText} / Table${numberText}` : `图${numberText} / Figure${numberText}`;
  const page = asset.page ? ` · 第 ${escapeHtml(asset.page)} 页` : "";
  const order = index ? `${index}、` : "";
  return `${order}${kind}${rawId ? ` · ${escapeHtml(rawId)}` : ""}${page}`;
}

function renderVisualAssets(paper) {
  const assets = (paper.visual_assets || [])
    .map((asset) => ({...asset, url: runAssetUrl(asset.asset_path)}))
    .filter((asset) => asset.url);
  if (!assets.length) return "";
  return `
    <section class="method-visuals">
      <div class="method-visuals-head">
        <h4>关键图表</h4>
        <small>从论文图表中自动挑选，补充说明研究框架与方法</small>
      </div>
      <div class="visual-asset-grid">
        ${assets.slice(0, 4).map((asset, index) => `
          <figure class="visual-asset-card">
            <a href="${escapeHtml(asset.url)}" target="_blank" rel="noopener">
              <img src="${escapeHtml(asset.url)}" alt="${escapeHtml(asset.caption || asset.id || "paper visual asset")}" loading="lazy">
            </a>
            <figcaption>
              <b>${visualAssetLabel(asset, index + 1)}</b>
              ${asset.caption ? `<span><em>原文说明 / Original caption</em>${inlineMarkdown(compactText(asset.caption, 220))}</span>` : ""}
            </figcaption>
          </figure>
        `).join("")}
      </div>
    </section>
  `;
}

function renderMethodStepCards(detail) {
  return `
    <div class="method-step-grid">
      ${detail.methodSteps.map((step, index) => `
        <article class="method-step-card">
          <div class="step-index">${index + 1}</div>
          <div class="method-step-main">
            <h4>${escapeHtml(step.name)}</h4>
            <div class="step-copy">${sectionPointList(step.detail, 3, 300)}</div>
          </div>
          ${(step.formulas || []).map((formula) => `
            <div class="step-formula">
              <small>${escapeHtml([formula.id, formula.name || "公式辅助"].filter(Boolean).join(" · "))}</small>
              <div class="latex-formula">${renderLatexFormula(formula.formula)}</div>
              ${formula.note ? markdownBlock(formula.note, "formula-note") : ""}
              ${formula.used_for ? markdownBlock(`用途：${formula.used_for}`, "formula-note") : ""}
              ${renderFormulaVariables(formula)}
            </div>
          `).join("")}
        </article>
      `).join("")}
    </div>
  `;
}

function renderDetailedSections(detail) {
  return `
    <section class="band">
      <div class="band-title"><h3>研究问题展开</h3><small>解释问题背景、核心 gap 和综述价值</small></div>
      ${markdownBlock(detail.sections.problemDetail)}
    </section>

    <section class="band">
      <div class="band-title"><h3>研究方法展开</h3><small>先看整体方法，再看流程拆解</small></div>
      <article class="method-overview">
        <h4>整体概括</h4>
        ${markdownBlock(detail.sections.methodOverview, "module-copy")}
      </article>
      <article class="method-overview">
        <h4>方法要素</h4>
        ${markdownBlock(detail.methodText, "module-copy")}
      </article>
      <div class="band-title sub-title"><h3>方法流程</h3><small>每一步都配合公式或变量解释</small></div>
      ${renderMethodStepCards(detail)}
    </section>

    <section class="band">
      <div class="band-title"><h3>主要结论展开</h3><small>解释结果对报价策略和市场机制的含义</small></div>
      ${markdownBlock(detail.sections.conclusionDetail)}
    </section>

    <section class="band">
      <div class="band-title"><h3>局限性与未来改进</h3><small>说明不足，并给出可继续推进的方向</small></div>
      ${markdownBlock(detail.sections.limitationDetail)}
    </section>
  `;
}

function renderDetailSectionCard(title, subtitle, content, asPoints = false) {
  return `
    <section class="band detail-section-card">
      <div class="detail-card-head">
        <h3>${escapeHtml(title)}</h3>
        <small>${escapeHtml(subtitle)}</small>
      </div>
      <div class="detail-card-body">${asPoints ? sectionPointList(content, 5, 300) : markdownBlock(content)}</div>
    </section>
  `;
}

function renderDetailedSectionsV2(detail) {
  return `
    ${renderDetailSectionCard("研究问题展开", "解释问题背景、核心 gap 和综述价值", detail.sections.problemDetail)}

    <section class="band method-detail-band">
      <div class="band-title"><h3>研究方法展开</h3><small>先看整体方法，再看流程拆解</small></div>
      <article class="method-overview">
        <h4>整体概括</h4>
        ${markdownBlock(detail.sections.methodOverview, "module-copy")}
      </article>
      <article class="method-overview">
        <h4>方法要素</h4>
        ${markdownBlock(detail.methodText, "module-copy")}
      </article>
      ${renderVisualAssets(currentPaper)}
      <div class="band-title sub-title"><h3>方法流程</h3><small>每一步都配合公式或变量解释</small></div>
      ${renderMethodStepCards(detail)}
    </section>

    ${renderDetailSectionCard("主要结论展开", "解释结果对报价策略和市场机制的含义", detail.sections.conclusionDetail, true)}

    ${renderDetailSectionCard("局限性与未来改进", "说明不足，并给出可继续推进的方向", detail.sections.limitationDetail, true)}
  `;
}

function renderFormulaBlock(items) {
  return `
    <div class="formula-grid">
      ${items.map((item) => `
        <article class="formula-card">
          <h4>${escapeHtml(item.name)}</h4>
          <div class="latex-formula">${renderLatexFormula(item.formula)}</div>
          ${markdownBlock(item.note, "formula-note")}
        </article>
      `).join("")}
    </div>
  `;
}

function reproStatusLabel(status) {
  const labels = {
    missing_pdf: "缺少 PDF",
    not_started: "可启动",
    prepared: "已初始化",
    audited: "已完成审计",
    model_spec_ready: "已抽取模型",
    workspace_ready: "工作区已生成"
  };
  return labels[status] || status || "未知";
}

function renderPaperReproductionPanel(paper, direction) {
  const repro = paper.reproduction || {};
  const scores = repro.scores || {};
  const counts = repro.model_spec_counts || {};
  const canRun = Boolean(activeRunId && repro.pdf_available);
  const links = Object.entries(repro.links || {}).filter(([, url]) => url);
  return `
    <section class="band paper-repro-band">
      <div class="band-title">
        <div>
          <h3>复现工具链</h3>
          <small>把当前单篇文件交给 repro_cli，生成审计、模型规范、数据模板和代码骨架</small>
        </div>
        <small>${escapeHtml(reproStatusLabel(repro.status))}</small>
      </div>
      <div class="paper-repro-layout">
        <div class="paper-repro-summary">
          <div class="metrics repro-mini-metrics">
            ${metric("PDF 状态", repro.pdf_available ? "已定位" : "未定位")}
            ${metric("参数项", counts.parameters || 0)}
            ${metric("约束项", counts.constraints || 0)}
            ${metric("工作区", repro.target_id ? "可打开" : "待生成")}
          </div>
          <div class="paper-repro-file">
            <span>复现目标</span>
            <b>${escapeHtml(repro.target_id || "待生成")}</b>
            <span>文件</span>
            <b>${escapeHtml(repro.pdf_name || "当前展示数据未包含可访问 PDF")}</b>
          </div>
          ${repro.blockers?.length ? `<div class="paper-repro-blockers"><h4>主要阻塞</h4>${listBlock(repro.blockers, 4)}</div>` : ""}
          <div class="repro-link-row">
            ${links.map(([key, url]) => `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(key.replace(/_/g, " "))}</a>`).join("")}
            ${repro.target_id ? `<a href="?view=reproduction#repro-${escapeHtml(repro.target_id)}">打开复现总览</a>` : ""}
          </div>
        </div>
        <div class="paper-repro-control">
          <div class="check-list repro-stage-options">
            <label><input type="checkbox" value="audit" checked> <span>论文拆解审计</span></label>
            <label><input type="checkbox" value="model-spec" checked> <span>模型参数抽取</span></label>
            <label><input type="checkbox" value="prepare-repro" checked> <span>生成数据模板与代码骨架</span></label>
          </div>
          <label class="check-row paper-repro-offline">
            <input id="reproOffline" type="checkbox" checked>
            <span>离线快速审计</span>
          </label>
          <button class="primary-button" id="startReproJob" type="button" ${canRun ? "" : "disabled"}>启动当前文件复现链路</button>
          <p class="job-state" id="reproJobState">${canRun ? "建议先离线生成结构化审计；需要更准确的模型抽取时再取消离线调用大模型。" : "当前页面没有可访问的 PDF，或处于静态预览模式。"}</p>
          <div class="job-progress hidden" id="reproJobProgress">
            <div class="progress-head">
              <div>
                <strong id="reproProgressStage">等待启动</strong>
                <small id="reproProgressTarget"></small>
              </div>
              <b id="reproProgressPercent">0%</b>
            </div>
            <div class="progress-bar"><span id="reproProgressBar"></span></div>
            <div class="progress-stats">
              <span><b id="reproProgressSteps">0/0</b><small>阶段</small></span>
              <span><b id="reproProgressScore">-</b><small>审计状态</small></span>
              <span><b id="reproProgressParams">0</b><small>参数项</small></span>
              <span><b id="reproProgressFiles">0</b><small>完整数据表</small></span>
            </div>
          </div>
        </div>
      </div>
    </section>
  `;
}

function bindPaperReproductionPanel(paper, direction) {
  const button = document.querySelector("#startReproJob");
  if (!button) return;
  button.addEventListener("click", async () => {
    if (!activeRunId) return;
    const stateEl = document.querySelector("#reproJobState");
    const stages = [...document.querySelectorAll(".repro-stage-options input:checked")].map((item) => item.value);
    button.disabled = true;
    if (stateEl) stateEl.textContent = "正在提交复现工具链任务...";
    const response = await fetch("/api/repro-jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        run_id: activeRunId,
        direction_id: direction.id,
        paper_id: paper.id,
        stages,
        offline: document.querySelector("#reproOffline")?.checked ?? true
      })
    });
    const job = await response.json();
    if (!response.ok) {
      if (stateEl) stateEl.textContent = job.error || "提交失败";
      button.disabled = false;
      return;
    }
    renderReproProgress(job);
    pollReproJob(job.id, paper, direction);
  });
}

function renderReproProgress(job) {
  const progress = job.progress || {};
  document.querySelector("#reproJobProgress")?.classList.remove("hidden");
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  const setText = (selector, value) => {
    const el = document.querySelector(selector);
    if (el) el.textContent = value;
  };
  setText("#reproProgressStage", progress.stage_label || job.status || "运行中");
  setText("#reproProgressTarget", progress.target_id || job.target_id || "");
  setText("#reproProgressPercent", `${percent}%`);
  const bar = document.querySelector("#reproProgressBar");
  if (bar) bar.style.width = `${percent}%`;
  setText("#reproProgressSteps", `${progress.completed_steps || 0}/${progress.total_steps || 0}`);
  setText("#reproProgressScore", progress.scores ? "已生成" : "-");
  setText("#reproProgressParams", progress.model_spec_counts?.parameters || 0);
  setText("#reproProgressFiles", progress.data_validation?.complete_files || 0);
  setText("#reproJobState", progress.stage_label || job.status || "");
}

async function refreshPaperReproStatus(paper, direction) {
  if (!activeRunId) return;
  const response = await fetch(`/api/reproduction/paper?run=${encodeURIComponent(activeRunId)}&direction=${encodeURIComponent(direction.id)}&paper=${encodeURIComponent(paper.id)}`);
  if (!response.ok) return;
  const payload = await response.json();
  paper.reproduction = payload.reproduction || paper.reproduction;
}

async function pollReproJob(jobId, paper, direction) {
  const response = await fetch(`/api/repro-jobs/${encodeURIComponent(jobId)}`);
  const job = await response.json();
  if (!response.ok) {
    const stateEl = document.querySelector("#reproJobState");
    if (stateEl) stateEl.textContent = job.error || "任务状态暂不可用";
    return;
  }
  renderReproProgress(job);
  if (job.status === "completed") {
    await refreshPaperReproStatus(paper, direction);
    renderPaper(direction.id, paper.id);
  } else if (job.status === "failed") {
    const stateEl = document.querySelector("#reproJobState");
    if (stateEl) stateEl.textContent = `失败：${job.error || "未知错误"}`;
    const button = document.querySelector("#startReproJob");
    if (button) button.disabled = false;
  } else {
    if (activeReproPollTimer) clearTimeout(activeReproPollTimer);
    activeReproPollTimer = setTimeout(() => pollReproJob(jobId, paper, direction), 2500);
  }
}

function renderPaper(directionId, paperId) {
  currentDirection = getDirection(directionId);
  currentPaper = getPaper(currentDirection, paperId);
  if (!currentPaper || !currentDirection) {
    renderOverview();
    return;
  }
  const detail = paperDetail(currentPaper, currentDirection);
  setHeader("paper", currentPaper.title_cn, "先用四个部分快速理解，再展开研究问题、方法流程、结论和局限。");
  canvas.innerHTML = `
    <section class="paper-hero">
      <div class="band">
        <div class="band-title">
          <div>
            <h3>${escapeHtml(currentPaper.title_cn)}</h3>
            <small>${escapeHtml(currentPaper.title)}</small>
          </div>
        </div>
        ${markdownBlock(detail.sections.coreSummary)}
      </div>
      <aside class="paper-meta">
        <div><span>作者</span><b>${escapeHtml(currentPaper.authors)}</b></div>
        <div><span>年份</span><b>${escapeHtml(currentPaper.year)}</b></div>
        <div><span>期刊</span><b>${escapeHtml(currentPaper.venue)}</b></div>
        <div><span>方向</span><b>${escapeHtml(currentDirection.name)}</b></div>
        ${renderOriginalLinkMeta(currentPaper)}
      </aside>
    </section>

    <section class="band">
      <div class="band-title"><h3>简略介绍</h3><small>每个部分控制在 300 字以内，保留完整表达</small></div>
      ${renderBriefSections(detail)}
    </section>

    ${renderKeywordRelatedSection(currentPaper)}

    ${renderPaperReproductionPanel(currentPaper, currentDirection)}

    ${renderDetailedSectionsV2(detail)}
  `;
  bindPaperReproductionPanel(currentPaper, currentDirection);
  typesetMath();
}

function addClause(logic = "and", text = "") {
  const row = document.createElement("div");
  row.className = "topic-row";
  row.innerHTML = `
    <select aria-label="条件逻辑">
      <option value="and"${logic === "and" ? " selected" : ""}>且</option>
      <option value="or"${logic === "or" ? " selected" : ""}>或</option>
      <option value="not"${logic === "not" ? " selected" : ""}>非</option>
    </select>
    <input type="text" value="${escapeHtml(text)}" placeholder="关键词或同义词，逗号分隔">
    <button class="icon-button remove-clause" type="button" title="删除条件" aria-label="删除条件">x</button>
  `;
  row.querySelector(".remove-clause").addEventListener("click", () => row.remove());
  topicClauses.appendChild(row);
}

function collectClauses() {
  return Array.from(topicClauses.querySelectorAll(".topic-row"))
    .map((row) => ({
      logic: row.querySelector("select").value,
      text: row.querySelector("input").value.trim()
    }))
    .filter((item) => item.text);
}

function syncModeControls() {
  const isLocal = document.querySelector("#mode")?.value === "pdf_only";
  document.querySelector("#localPdfControls")?.classList.toggle("hidden", !isLocal);
  document.querySelector("#maxResultsField")?.classList.toggle("hidden", isLocal);
  document.querySelector("#maxResults")?.toggleAttribute("disabled", isLocal);
  document.querySelector(".year-range-controls")?.classList.toggle("hidden", isLocal);
}

function numberText(value) {
  if (value === "" || value === null || value === undefined) return "0";
  return String(value);
}

function trackedJobs() {
  try {
    const rows = JSON.parse(localStorage.getItem(JOB_STORE_KEY) || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

function saveTrackedJobs(rows) {
  localStorage.setItem(JOB_STORE_KEY, JSON.stringify(rows.slice(-15)));
}

function createClientJobId() {
  const random = Math.random().toString(36).slice(2, 8);
  return `web_${Date.now()}_${random}`;
}

function rememberJob(job) {
  const progress = job.progress || {};
  const runId = progress.run_id || job.run_id || "";
  if (!job.id) return;
  const rows = trackedJobs().filter((item) => item.id !== job.id && item.run_id !== runId);
  rows.push({
    id: job.id,
    run_id: runId,
    topic: job.topic || "",
    status: job.status || "",
    updated_at: new Date().toISOString()
  });
  saveTrackedJobs(rows);
}

function updateTrackedJob(job) {
  const progress = job.progress || {};
  const runId = progress.run_id || "";
  const rows = trackedJobs().map((item) => {
    if (item.id === job.id || (runId && item.run_id === runId)) {
      return {...item, status: job.status || item.status, updated_at: new Date().toISOString()};
    }
    return item;
  });
  saveTrackedJobs(rows);
}

async function fetchTrackedJob(record) {
  try {
    let response = null;
    if (record.id && !String(record.id).startsWith("run:")) {
      response = await fetch(`/api/jobs/${encodeURIComponent(record.id)}`);
      if (response.ok) return response.json();
    }
    if (record.run_id) {
      response = await fetch(`/api/jobs/by-run/${encodeURIComponent(record.run_id)}`);
      if (response.ok) return response.json();
    }
  } catch {
    return null;
  }
  return null;
}

function shouldDisplayJob(job) {
  const runId = job.progress?.run_id || "";
  return visibleJobId === job.id || Boolean(activeRunId && runId === activeRunId);
}

function schedulePoll(id) {
  if (activePollTimer) clearTimeout(activePollTimer);
  activePollTimer = setTimeout(() => pollJob(id), 5000);
}

function formatDuration(seconds) {
  if (seconds === "" || seconds === null || seconds === undefined) return "";
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "";
  const total = Math.round(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function renderProgress(job) {
  const progress = job.progress || {};
  progressPanel?.classList.remove("hidden");
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  if (progressStage) progressStage.textContent = progress.stage_label || job.status || "后台运行中";
  if (progressRunId) {
    const target = progress.target_papers ? `目标 ${progress.target_papers} 篇` : "";
    progressRunId.textContent = [progress.run_id, target].filter(Boolean).join(" · ");
  }
  if (progressPercent) progressPercent.textContent = `${percent}%`;
  if (progressBar) progressBar.style.width = `${percent}%`;
  if (progressSearched) progressSearched.textContent = numberText(progress.searched_papers);
  if (progressDownloaded) progressDownloaded.textContent = numberText(progress.downloaded_papers);
  if (progressAnalyzed) progressAnalyzed.textContent = numberText(progress.analyzed_papers);
  if (progressFiltered) progressFiltered.textContent = numberText(progress.reviewed_directions || 0);
  if (progressSteps) {
    const rows = [];
    for (const step of progress.steps || []) {
      const status = step.status === "running" ? "运行中" : step.status === "completed" ? "完成" : step.status === "failed" ? "失败" : step.status === "pending" ? "待开始" : step.status || "";
      const duration = formatDuration(step.elapsed_seconds);
      const statusText = [status, duration].filter(Boolean).join(" · ");
      rows.push(`<li class="${step.status === "running" ? "current" : ""}"><b>${escapeHtml(step.code || "")}</b><span>${escapeHtml(step.label || "")}</span><em>${escapeHtml(statusText)}</em></li>`);
    }
    progressSteps.innerHTML = rows.length ? rows.join("") : '<li><b>-</b><span>等待阶段记录</span><em>待开始</em></li>';
  }
  if (progressNotes) {
    const notes = progress.notes || [];
    progressNotes.innerHTML = notes.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    progressNotes.classList.toggle("hidden", !notes.length);
  }
}

async function pollJob(id) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(id)}`);
  const job = await response.json();
  if (!response.ok) {
    const retryCount = missingJobRetries[id] || 0;
    const isPendingClientJob = String(id || "").startsWith("web_") && retryCount < 24;
    if (isPendingClientJob) {
      missingJobRetries[id] = retryCount + 1;
      state.textContent = "后台任务正在启动，页面可继续切换查看。";
      schedulePoll(id);
      return;
    }
    state.textContent = job.error || "任务状态暂不可用";
    startButton.disabled = false;
    return;
  }
  delete missingJobRetries[id];
  rememberJob(job);
  updateTrackedJob(job);
  if (shouldDisplayJob(job)) renderProgress(job);
  const label = job.progress?.stage_label || job.status;
  if (shouldDisplayJob(job)) {
    state.textContent = `${label}${job.returncode !== undefined ? ` · code ${job.returncode}` : ""}`;
  }
  if (job.status === "completed") {
    startButton.disabled = false;
    const runId = job.progress?.run_id;
    if (shouldDisplayJob(job)) {
      state.innerHTML = runId
        ? `任务完成，<a href="/?run=${encodeURIComponent(runId)}">打开最新展示结果</a>`
        : "任务完成，可在已有运行中打开结果。";
    }
  } else if (job.status === "failed") {
    if (shouldDisplayJob(job)) state.textContent = `失败：${job.error || "未知错误"}`;
    startButton.disabled = false;
  } else {
    if (shouldDisplayJob(job)) startButton.disabled = true;
    schedulePoll(id);
  }
}

async function resumeTrackedJobForCurrentRun() {
  const records = trackedJobs().filter((item) => item && item.id);
  if (activeRunId) {
    const activeRecords = records.filter((item) => item.run_id === activeRunId).reverse();
    const activeJob = activeRecords.length
      ? await fetchTrackedJob(activeRecords[0])
      : await fetchTrackedJob({id: `run:${activeRunId}`, run_id: activeRunId});
    if (!activeJob || activeJob.progress?.run_id !== activeRunId) return;
    updateTrackedJob(activeJob);
    visibleJobId = activeJob.id;
    renderProgress(activeJob);
    if (activeJob.status === "completed") {
      state.textContent = activeJob.progress?.stage_label || "已完成";
      startButton.disabled = false;
    } else if (activeJob.status === "failed") {
      state.textContent = `失败：${activeJob.error || "未知错误"}`;
      startButton.disabled = false;
    } else {
      state.textContent = `${activeJob.progress?.stage_label || activeJob.status}`;
      startButton.disabled = !String(activeJob.id).startsWith("run:");
      if (!String(activeJob.id).startsWith("run:")) pollJob(activeJob.id);
    }
    return;
  }
  const activeMatches = activeRunId ? records.filter((item) => item.run_id === activeRunId) : [];
  const activeRunningMatches = activeMatches.filter((item) => !isTerminalJobStatus(item.status));
  const runningMatches = records.filter((item) => !isTerminalJobStatus(item.status));
  const preferred = activeRunningMatches.length ? activeRunningMatches : runningMatches.length ? runningMatches : activeMatches;
  for (const record of preferred.reverse()) {
    const job = await fetchTrackedJob(record);
    if (!job || !job.progress?.run_id) continue;
    updateTrackedJob(job);
    visibleJobId = job.id;
    renderProgress(job);
    if (job.status === "completed") {
      state.innerHTML = `后台任务已完成，<a href="/?run=${encodeURIComponent(job.progress.run_id)}">打开结果</a>`;
      startButton.disabled = false;
    } else {
      state.textContent = `${job.progress.stage_label || job.status}`;
      startButton.disabled = !String(job.id).startsWith("run:");
    }
    if (!isTerminalJobStatus(job.status) && !String(job.id).startsWith("run:")) {
      pollJob(job.id);
    }
    break;
  }
}

function setWorkbenchMode(mode) {
  literatureViewButton?.classList.toggle("active", mode !== "reproduction");
  reproductionViewButton?.classList.toggle("active", mode === "reproduction");
  if (queryMetaBox) queryMetaBox.classList.toggle("hidden", mode === "reproduction");
}

function valueOrDash(value) {
  if (value === null || value === undefined || value === "") return "-";
  return value;
}

function listBlock(items, maxItems = 6) {
  const rows = asArray(items).map((item) => cleanText(item)).filter(Boolean).slice(0, maxItems);
  return `<ul class="point-list">${(rows.length ? rows : ["暂无结构化记录"]).map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`;
}

function renderCsvPreview(table, maxRows = 8) {
  const headers = asArray(table?.headers).filter(Boolean);
  const rows = asArray(table?.rows).slice(0, maxRows);
  if (!headers.length || !rows.length) return `<p class="summary-text">暂无表格数据。</p>`;
  return `
    <div class="paper-table repro-table">
      <table>
        <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>${headers.map((header) => `<td>${inlineMarkdown(compactText(row?.[header], 180))}</td>`).join("")}</tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderCheckRows(rows, titleField, detailField, maxRows = 8) {
  const items = asArray(rows).slice(0, maxRows);
  if (!items.length) return `<p class="summary-text">暂无检查项。</p>`;
  return `
    <div class="repro-check-list">
      ${items.map((item) => {
        const status = cleanText(item.status || item.reproduction_status || item.availability || "");
        return `
          <article>
            <div>
              <b>${escapeHtml(item[titleField] || item.item || item.component || item.target || "检查项")}</b>
              ${status ? `<span class="status-pill">${escapeHtml(status)}</span>` : ""}
            </div>
            <p>${escapeHtml(compactText(item[detailField] || item.evidence || item.notes || "", 260))}</p>
          </article>
        `;
      }).join("")}
    </div>
  `;
}

function renderModelSpec(target) {
  const spec = target.model_spec || {};
  const blocks = [
    ["集合", spec.sets],
    ["关键参数", spec.parameters],
    ["决策变量", spec.variables],
    ["约束结构", spec.constraints],
    ["不确定性", spec.uncertainty],
    ["实现提示", spec.implementation_notes]
  ];
  return `
    <div class="module-grid repro-model-grid">
      <article class="module repro-objective">
        <div class="module-head"><h4>目标函数</h4><small>model_spec</small></div>
        <p>${escapeHtml(compactText(spec.objective || "暂无目标函数描述", 360))}</p>
      </article>
      ${blocks.map(([title, items]) => `
        <article class="module">
          <div class="module-head"><h4>${escapeHtml(title)}</h4><small>${asArray(items).length} 项</small></div>
          ${listBlock(items, 5)}
        </article>
      `).join("")}
    </div>
  `;
}

function renderDataTables(target) {
  const tables = asArray(target.case_data?.tables);
  if (!tables.length) return `<p class="summary-text">暂无数据表摘要。</p>`;
  return `
    <div class="repro-data-grid">
      ${tables.slice(0, 12).map((table) => `
        <article>
          <b>${escapeHtml(table.name)}</b>
          <span>${escapeHtml(valueOrDash(table.rows))} 行</span>
          <small>${asArray(table.columns).map((item) => escapeHtml(item)).join(" / ")}</small>
        </article>
      `).join("")}
    </div>
  `;
}

function renderMaterialCards(files, emptyText = "暂无可打开材料。") {
  const items = asArray(files).filter((item) => item.url);
  if (!items.length) return `<p class="summary-text">${escapeHtml(emptyText)}</p>`;
  return `
    <div class="repro-material-grid">
      ${items.map((file) => `
        <a class="repro-material-card" href="${escapeHtml(file.url)}" data-file-label="${escapeHtml(file.label || file.name)}">
          <span>${escapeHtml(file.kind || "file")}</span>
          <b>${escapeHtml(file.label || file.name)}</b>
          <small>${escapeHtml(file.relative || file.name)}</small>
          ${asArray(file.columns).length ? `<em>${asArray(file.columns).slice(0, 4).map((item) => escapeHtml(item)).join(" / ")}</em>` : ""}
        </a>
      `).join("")}
    </div>
  `;
}

function classifyDataSource(row) {
  const source = cleanText(row.source_hint || "");
  const status = cleanText(row.reproduction_status || row.availability || "");
  const notes = cleanText(row.notes || "");
  const text = `${source} ${status} ${notes}`.toLowerCase();
  if (/matpower|pglib|ieee|open source|public base case|github|code/.test(text)) {
    return {
      label: "开源基准/开源代码",
      detail: "可从公开算例、开源代码库或标准测试系统获取基础数据，再记录论文改动。",
    };
  }
  if (/assumption|needs assumption/.test(text)) {
    return {
      label: "合理构造/待确认假设",
      detail: "论文未给出完整数值，需要根据公开基准、工程常识或用户确认进行构造。",
    };
  }
  if (/extraction|needs extraction|available|rebuildable/.test(text) && !/partially/.test(text)) {
    return {
      label: "论文抽取/论文规则重建",
      detail: "来自正文、图表、公式或参数说明，可按论文规则整理成结构化表。",
    };
  }
  if (/cited|benchmark|reference/.test(text)) {
    return {
      label: "引用基准/可追踪资料",
      detail: "需要追踪论文引用的数据集、算例或附录，适合作为可核验来源。",
    };
  }
  if (/paper/.test(text)) {
    return {
      label: "论文抽取/论文规则重建",
      detail: "来自正文、图表、公式或参数说明，可按论文规则整理成结构化表。",
    };
  }
  if (/partially|needs/.test(text)) {
    return {
      label: "合理构造/待确认假设",
      detail: "论文未给出完整数值，需要根据公开基准、工程常识或用户确认进行构造。",
    };
  }
  return {
    label: "待确认来源",
    detail: "需要继续检索或与用户确认来源、单位和可替代数据。",
  };
}

function renderDataSourceRegistry(target) {
  const rows = asArray(target.dataset_registry?.rows);
  if (!rows.length) return `<p class="summary-text">暂无数据来源登记表。</p>`;
  const registryFile = asArray(target.materials?.processed_files).find((file) => file.name === "dataset_registry.csv" || String(file.relative || "").endsWith("dataset_registry.csv"));
  if (target.id === "nasri_2016_ac_uc_benders") {
    const layeredRows = [
      ["网络线路容量", "原文 Table I 转录", "transcribe_nasri_tables.py", "覆盖部分线路容量"],
      ["机组参数", "原文 Table II 转录", "transcribe_nasri_tables.py", "生成机组上下限、成本、备用能力"],
      ["负荷因子", "原文 Table III 转录", "transcribe_nasri_tables.py", "生成 24 小时负荷因子"],
      ["场景概率", "原文 Table IV 转录", "transcribe_nasri_tables.py", "生成 40 个场景概率，概率和为 1"],
      ["RTS 基础负荷", "公开 RTS / MATPOWER case24_ieee_rts.m", "fill_bus_load_fractions.py", "Pd/Qd 乘以负荷因子"],
      ["风电时序", "原文 Fig. 3 无机器可读数据", "generate_surrogate_wind_profiles.py", "生成 40 场景 x 24 小时 x 2 风场"],
      ["案例调整", "原文实验设定与已转录表格", "apply_nasri_case_adjustments.py", "统一写入 data/ 下 CSV 数据层"],
    ];
    return `
      <div class="data-source-registry data-source-layered">
        <div class="module-head">
          <h4>2.1 数据来源分层</h4>
          <small>先区分原文可转录表格、公开 RTS/MATPOWER 基准和无原始数据的替代构造，再统一进入 CSV 数据层</small>
        </div>
        <div class="data-layer-diagram" aria-label="数据来源分层流程图">
          <svg viewBox="0 0 920 290" role="img">
            <defs>
              <marker id="dataArrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L8,4 L0,8 Z"></path>
              </marker>
            </defs>
            <g class="data-flow-box">
              <rect x="40" y="18" width="170" height="58"></rect>
              <text x="125" y="52">原文表格</text>
              <rect x="18" y="128" width="214" height="58"></rect>
              <text x="125" y="163">可直接转录数据</text>
            </g>
            <g class="data-flow-box">
              <rect x="324" y="18" width="272" height="58"></rect>
              <text x="460" y="52">公开 RTS / MATPOWER</text>
              <rect x="344" y="128" width="232" height="58"></rect>
              <text x="460" y="163">基础网络与负荷分布</text>
            </g>
            <g class="data-flow-box">
              <rect x="692" y="18" width="214" height="58"></rect>
              <text x="799" y="52">原文图形但无原始数据</text>
              <rect x="708" y="128" width="182" height="58"></rect>
              <text x="799" y="163">合成替代数据</text>
            </g>
            <g class="data-flow-box data-flow-merge">
              <rect x="345" y="224" width="230" height="58"></rect>
              <text x="460" y="258">统一 CSV 数据层</text>
            </g>
            <g class="data-flow-lines">
              <path d="M125 76 L125 122"></path>
              <path d="M460 76 L460 122"></path>
              <path d="M799 76 L799 122"></path>
              <path d="M125 186 C125 218, 265 218, 338 244"></path>
              <path d="M460 186 L460 218"></path>
              <path d="M799 186 C799 218, 655 218, 582 244"></path>
            </g>
          </svg>
        </div>
        <div class="paper-table repro-table data-layer-table">
          <table>
            <thead>
              <tr>
                <th>数据类型</th>
                <th>获取方式</th>
                <th>转换脚本</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              ${layeredRows.map((row) => `
                <tr>
                  <td>${escapeHtml(row[0])}</td>
                  <td>${escapeHtml(row[1])}</td>
                  <td><code>${escapeHtml(row[2])}</code></td>
                  <td>${escapeHtml(row[3])}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
        ${registryFile?.url ? `<div class="repro-stage-actions"><a href="${escapeHtml(registryFile.url)}" data-file-label="${escapeHtml(registryFile.label || registryFile.name)}">打开完整数据来源表</a></div>` : ""}
      </div>
    `;
  }
  return `
    <div class="data-source-registry">
      <div class="module-head">
        <h4>数据来源表格</h4>
        <small>区分开源基准、论文抽取、合理构造和待确认假设</small>
      </div>
      <div class="paper-table repro-table">
        <table>
          <thead>
            <tr>
              <th>数据项</th>
              <th>数据类型</th>
              <th>来源分类</th>
              <th>来源线索</th>
              <th>复现说明</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => {
              const sourceType = classifyDataSource(row);
              return `
                <tr>
                  <td>${escapeHtml(dataPreparationMeta(row).title)}</td>
                  <td>${escapeHtml(dataPreparationMeta(row).type)}</td>
                  <td><span class="source-kind-pill">${escapeHtml(sourceType.label)}</span><small>${escapeHtml(sourceType.detail)}</small></td>
                  <td>${escapeHtml(row.source_hint || "待确认")}</td>
                  <td>${escapeHtml([dataStatusCn(row), row.notes].filter(Boolean).join("；"))}</td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
      ${registryFile?.url ? `<div class="repro-stage-actions"><a href="${escapeHtml(registryFile.url)}" data-file-label="${escapeHtml(registryFile.label || registryFile.name)}">打开完整数据来源表</a></div>` : ""}
    </div>
  `;
}

function renderMaterialHub(target) {
  const materials = target.materials || {};
  return `
    <section class="band">
      <div class="band-title">
        <div>
          <h3>可打开的数据与参数材料</h3>
          <small>直接查看下载好的 CSV、结构化参数、配置、校验报告和处理产物</small>
        </div>
      </div>
      <div class="repro-material-columns">
        <article>
          <div class="module-head"><h4>数据文件</h4><small>${asArray(materials.data_files).length} 个</small></div>
          ${renderMaterialCards(materials.data_files, "暂无数据文件。")}
        </article>
        <article>
          <div class="module-head"><h4>参数与配置</h4><small>${asArray(materials.parameter_files).length} 个</small></div>
          ${renderMaterialCards(materials.parameter_files, "暂无参数配置文件。")}
        </article>
        <article>
          <div class="module-head"><h4>处理后产物</h4><small>${asArray(materials.processed_files).length} 个</small></div>
          ${renderMaterialCards(materials.processed_files, "暂无处理产物。")}
        </article>
        <article>
          <div class="module-head"><h4>对话生成产物</h4><small>${asArray(materials.generated_files).length} 个</small></div>
          ${renderMaterialCards(materials.generated_files, "运行阶段二示例对话后，这里会出现生成的 CSV、脚本和说明文件。")}
        </article>
      </div>
      ${renderDataSourceRegistry(target)}
    </section>
  `;
}

function ensureFilePreviewModal() {
  let modal = document.querySelector("#filePreviewModal");
  if (modal) return modal;
  modal = document.createElement("div");
  modal.id = "filePreviewModal";
  modal.className = "file-preview-modal hidden";
  modal.innerHTML = `
    <div class="file-preview-backdrop" data-file-preview-close="true"></div>
    <section class="file-preview-panel" role="dialog" aria-modal="true" aria-label="文件预览">
      <header>
        <div>
          <h3 id="filePreviewTitle">文件预览</h3>
          <small id="filePreviewPath"></small>
        </div>
        <div class="file-preview-actions">
          <a id="filePreviewOpen" href="#" target="_blank" rel="noopener">新页面打开</a>
          <button type="button" data-file-preview-close="true">关闭</button>
        </div>
      </header>
      <div class="file-preview-body" id="filePreviewBody"></div>
    </section>
  `;
  document.body.appendChild(modal);
  modal.addEventListener("click", (event) => {
    if (event.target.closest("[data-file-preview-close]")) {
      modal.classList.add("hidden");
    }
  });
  return modal;
}

function renderPreviewContent(url, text) {
  const cleanUrl = String(url || "").split("?")[0].toLowerCase();
  if (cleanUrl.endsWith(".md")) {
    return `<div class="file-preview-markdown">${renderMarkdown(text)}</div>`;
  }
  if (cleanUrl.endsWith(".json")) {
    try {
      return `<pre><code>${escapeHtml(JSON.stringify(JSON.parse(text), null, 2))}</code></pre>`;
    } catch {
      return `<pre><code>${escapeHtml(text)}</code></pre>`;
    }
  }
  return `<pre><code>${escapeHtml(text)}</code></pre>`;
}

async function openFilePreview(url, label) {
  const modal = ensureFilePreviewModal();
  const title = modal.querySelector("#filePreviewTitle");
  const path = modal.querySelector("#filePreviewPath");
  const open = modal.querySelector("#filePreviewOpen");
  const body = modal.querySelector("#filePreviewBody");
  title.textContent = label || "文件预览";
  path.textContent = url;
  open.href = url;
  body.innerHTML = `<p class="summary-text">正在读取文件...</p>`;
  modal.classList.remove("hidden");
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    body.innerHTML = renderPreviewContent(url, text);
  } catch (error) {
    body.innerHTML = `<p class="summary-text">文件读取失败：${escapeHtml(error.message || String(error))}</p>`;
  }
}

function renderStageVerdict(target) {
  const validation = target.data_validation || {};
  const completeFiles = Number(validation.complete_files || 0);
  const emptyFiles = Number(validation.empty_files || 0);
  const missingFiles = Number(validation.missing_files || 0);
  const hasModelSpec = asArray(target.model_spec?.parameters).length || asArray(target.model_spec?.constraints).length;
  const hasEvidence = asArray(target.evidence_snippets).length > 0;
  const hasEnvironment = asArray(target.materials?.parameter_files).some((file) => String(file.name || "").includes("solver_config"));
  const hasDialogue = asArray(target.llm_prompt_assets).length > 0 || target.data_completion?.prompt;
  return `
    <section class="band repro-stage-band">
      <div class="band-title">
        <div>
          <h3>两阶段复现架构</h3>
          <small>阶段一拆解论文，阶段二进入个性化生成式协作</small>
        </div>
      </div>
      <div class="repro-stage-grid">
        <article class="repro-stage-card ${hasEvidence || hasModelSpec ? "is-ready" : ""}">
          <span>阶段一</span>
          <h4>论文拆解与工作流确认</h4>
          <p>系统从原文中抽取数据来源、数据类型、模型结构和环境要求，形成固定工具链可以稳定处理的准备清单。</p>
          <div class="repro-stage-actions">
            ${target.links?.model_spec_md ? `<a href="${escapeHtml(target.links.model_spec_md)}" target="_blank" rel="noopener">模型规范</a>` : ""}
            ${target.links?.source_trace_md ? `<a href="${escapeHtml(target.links.source_trace_md)}" target="_blank" rel="noopener">打开来源追踪</a>` : ""}
          </div>
        </article>
        <article class="repro-stage-card ${hasDialogue ? "is-pending" : "is-blocked"}">
          <span>阶段二</span>
          <h4>大模型多轮协作工作台</h4>
          <p>用户可围绕下载数据、编写脚本、检查图表和解释差距继续追问，大模型根据当前工作区文件给出下一步操作。</p>
          <ul class="repro-stage-reasons">
            <li>当前完整数据表：${escapeHtml(valueOrDash(completeFiles))} 个；待补齐或异常数据表：${escapeHtml(emptyFiles + missingFiles)} 个。</li>
            <li>环境配置${hasEnvironment ? "已生成" : "待确认"}；图表产物 ${escapeHtml(asArray(target.figures).length)} 个。</li>
          </ul>
          <div class="repro-stage-actions">
            <a href="#stageTwoWorkspace">进入对话工作台</a>
            ${asArray(target.figures).length ? `<a href="#reproFigures">查看图表</a>` : ""}
          </div>
        </article>
      </div>
    </section>
  `;
}

function collectTargetFiles(target) {
  const materials = target.materials || {};
  const groups = [
    ["data", "数据文件", materials.data_files],
    ["params", "参数与配置", materials.parameter_files],
    ["processed", "处理产物", materials.processed_files],
    ["generated", "对话生成产物", materials.generated_files],
    ["code", "脚本文件", target.code_artifacts],
    ["prompt", "提示词与技能文档", target.llm_prompt_assets],
  ];
  return groups.map(([id, label, files]) => ({
    id,
    label,
    files: asArray(files).filter((file) => file?.url).map((file) => ({
      ...file,
      cacheKey: `${target.id}:${id}:${file.relative || file.name || file.url}`,
      label: file.label || file.role || file.name || file.relative || "文件",
    })),
  }));
}

function targetProgress(target) {
  const validation = target.data_validation || {};
  const dataReady = Number(validation.complete_files || 0) > 0 && Number(validation.empty_files || 0) === 0 && Number(validation.missing_files || 0) === 0;
  const steps = [
    ["文献拆解", asArray(target.evidence_snippets).length > 0 || asArray(target.data_check).length > 0],
    ["数据准备", dataReady],
    ["模型结构", Boolean(target.model_spec?.objective || asArray(target.model_spec?.parameters).length)],
    ["环境配置", asArray(target.materials?.parameter_files).some((file) => /solver_config|experiment_matrix|assumptions/.test(String(file.name || file.relative || "")))],
    ["脚本草稿", asArray(target.code_artifacts).some((file) => String(file.group || "") === "generated")],
    ["图表输出", asArray(target.figures).length > 0],
  ];
  const completed = steps.filter(([, ready]) => ready).length;
  const percent = Math.round((completed / steps.length) * 100);
  return {steps, completed, percent, dataReady};
}

function renderReproductionProgress(target) {
  const progress = targetProgress(target);
  return `
    <section class="band repro-progress-band">
      <div class="band-title">
        <div>
          <h3>动态复现进度</h3>
          <small>根据当前工作区文件、数据校验和图表产物实时汇总</small>
        </div>
        <b class="repro-progress-percent">${progress.percent}%</b>
      </div>
      <div class="progress-bar repro-progress-bar"><span style="width:${progress.percent}%"></span></div>
      <div class="repro-progress-steps">
        ${progress.steps.map(([label, ready]) => `
          <span class="${ready ? "done" : "pending"}"><b>${ready ? "已完成" : "待推进"}</b>${escapeHtml(label)}</span>
        `).join("")}
      </div>
    </section>
  `;
}

const DATA_PREPARATION_CN = {
  "modified ieee 118-bus network": {
    title: "修改后的 IEEE 118 节点系统",
    type: "网络拓扑与节点负荷",
    summary: "需要准备节点、线路、基准容量、负荷分配和论文中对标准算例的改动说明。",
  },
  "uc generator parameters": {
    title: "机组组合参数",
    type: "常规机组与成本数据",
    summary: "需要准备机组出力上下限、爬坡、启停、最小开停机时间和分段成本曲线。",
  },
  "wind farms": {
    title: "风电场信息",
    type: "新能源场站数据",
    summary: "需要确认风场数量、接入节点、装机容量和论文使用的风电情景设定。",
  },
  "load profile": {
    title: "负荷曲线",
    type: "时间序列数据",
    summary: "需要准备 24 小时或论文实验周期内的系统负荷，并记录缩放基准和单位。",
  },
  "uncertainty set": {
    title: "不确定性集合",
    type: "模型参数",
    summary: "需要整理负荷、风电或市场价格的不确定性边界、情景概率和鲁棒/随机建模假设。",
  },
  "battery parameters": {
    title: "储能参数",
    type: "设备参数",
    summary: "需要准备容量、功率、效率、SOC 上下限、初始 SOC 和退化成本参数。",
  },
  "market prices": {
    title: "市场价格",
    type: "市场时间序列",
    summary: "需要准备能量、备用、调频容量或调频里程等价格序列，并记录市场来源。",
  },
};

function dataPreparationFallback(row) {
  const item = cleanText(row.item || "数据项");
  const type = cleanText(row.type || "数据");
  const source = cleanText(row.source_hint || row.notes || "");
  return {
    title: item,
    type,
    summary: source ? `需要围绕“${item}”准备${type}，来源线索为：${source}。` : `需要准备“${item}”相关${type}，并记录来源、单位和假设。`,
  };
}

function dataPreparationMeta(row) {
  const key = cleanText(row.item || "").toLowerCase();
  return DATA_PREPARATION_CN[key] || dataPreparationFallback(row);
}

function dataStatusCn(row) {
  const text = cleanText(row.reproduction_status || row.availability || "");
  const map = {
    "needs tracing": "需追踪来源",
    "needs assumption": "需补充假设",
    "needs extraction": "需从论文抽取",
    "rebuildable": "可重建",
    "public base case": "有公开基准",
    "partially available": "部分可得",
    "available": "已明确",
  };
  return map[text.toLowerCase()] || text || "待确认";
}

function renderSourceTypeList(target) {
  const rows = asArray(target.dataset_registry?.rows).slice(0, 8);
  if (!rows.length) return listBlock(target.data_check, 5);
  return `
    <div class="repro-source-list">
      ${rows.map((row) => {
        const meta = dataPreparationMeta(row);
        return `
        <article>
          <div>
            <b>${escapeHtml(meta.title)}</b>
            <span>${escapeHtml(meta.type)}</span>
          </div>
          <p>${escapeHtml(meta.summary)}</p>
          <small>${escapeHtml(dataStatusCn(row))}</small>
        </article>
      `;}).join("")}
    </div>
  `;
}

function renderModelEnvironmentBrief(target) {
  const spec = target.model_spec || {};
  const envFiles = asArray(target.materials?.parameter_files).filter((file) => /solver|experiment|assumption|config/.test(String(file.name || file.relative || "")));
  return `
    <div class="stage-one-brief-grid">
      <article>
        <div class="module-head"><h4>模型类型与目标</h4><small>来自模型规范抽取</small></div>
        <p>${escapeHtml(compactText(spec.objective || "待从原文目标函数段落确认。", 320))}</p>
        ${listBlock([...(spec.variables || []).slice(0, 3), ...(spec.constraints || []).slice(0, 3)], 6)}
      </article>
      <article>
        <div class="module-head"><h4>环境配置</h4><small>求解器、实验矩阵、假设文件</small></div>
        ${envFiles.length ? renderMaterialCards(envFiles, "暂无环境配置文件。") : `<p class="summary-text">待配置 Python 环境、优化求解器和实验参数文件。</p>`}
      </article>
    </div>
  `;
}

function renderStageOneExtraction(target) {
  return `
    <section class="band stage-one-extraction">
      <div class="band-title">
        <div>
          <h3>阶段一：论文拆解结果</h3>
          <small>将论文中的复现条件整理成中文准备清单，重点说明数据、模型和环境</small>
        </div>
      </div>
      <div class="stage-one-summary">
        <article>
          <h4>数据准备清单</h4>
          ${renderSourceTypeList(target)}
        </article>
        <article>
          <h4>模型与环境</h4>
          ${renderModelEnvironmentBrief(target)}
        </article>
      </div>
    </section>
  `;
}

function renderPromptChoiceCards(target) {
  const assets = asArray(target.llm_prompt_assets);
  const dataPrompt = target.data_completion?.prompt || "";
  return `
    <section class="band prompt-template-band">
      <div class="band-title">
        <div>
          <h3>可用提示词模板</h3>
          <small>以下是可用的提示词模板，可按当前任务选择后交给大模型继续细化</small>
        </div>
      </div>
      <div class="prompt-choice-grid">
        ${dataPrompt ? `
          <button class="prompt-choice-card prompt-open-button" type="button" data-prompt-kind="data-completion">
            <span>模板 1</span>
            <b>数据补齐协作提示词</b>
            <small>围绕空数据表、字段来源、单位和人工确认问题继续追问。</small>
            <em>打开模板</em>
          </button>
        ` : ""}
        ${assets.slice(0, 6).map((asset, index) => `
          <a class="prompt-choice-card prompt-asset-card" href="${escapeHtml(asset.url)}" data-file-label="${escapeHtml(asset.label || asset.name)}">
            <span>模板 ${dataPrompt ? index + 2 : index + 1}</span>
            <b>${escapeHtml(asset.label || asset.name)}</b>
            <small>${escapeHtml(asset.relative || "")}</small>
            <em>预览文件</em>
          </a>
        `).join("")}
      </div>
    </section>
    <textarea class="hidden" id="dataCompletionPrompt">${escapeHtml(dataPrompt)}</textarea>
  `;
}

function renderWorkFileCache(target) {
  const groups = collectTargetFiles(target);
  const materials = target.materials || {};
  const cacheStats = [
    ["数据文件", asArray(materials.data_files).length],
    ["参数配置", asArray(materials.parameter_files).length],
    ["处理产物", asArray(materials.processed_files).length],
    ["对话产物", asArray(materials.generated_files).length],
  ];
  return `
    <section class="band work-cache-band">
      <div class="band-title">
        <div>
          <h3>阶段二产出与工作文件缓存</h3>
          <small>把数据、参数、处理产物和多轮对话生成文件合并管理，按需放入本轮大模型上下文</small>
        </div>
        <div class="work-cache-actions">
          <button class="ghost-button" type="button" data-cache-action="select-data">选择数据</button>
          <button class="ghost-button" type="button" data-cache-action="clear">清空</button>
        </div>
      </div>
      <div class="work-cache-layout" data-cache-target="${escapeHtml(target.id)}">
        <aside class="work-cache-tree">
          ${groups.map((group) => `
            <details ${group.id === "data" ? "open" : ""}>
              <summary>${escapeHtml(group.label)} <small>${group.files.length}</small></summary>
              ${group.files.length ? group.files.map((file) => `
                <label>
                  <input type="checkbox" class="work-cache-check" data-cache-key="${escapeHtml(file.cacheKey)}" data-cache-url="${escapeHtml(file.url)}" data-cache-label="${escapeHtml(file.label)}">
                  <span>${escapeHtml(compactText(file.label, 46))}</span>
                </label>
              `).join("") : `<p class="summary-text">暂无文件</p>`}
            </details>
          `).join("")}
        </aside>
        <div class="work-cache-panel">
          <div class="work-cache-summary-grid">
            ${cacheStats.map(([label, count]) => `
              <span><b>${escapeHtml(count)}</b>${escapeHtml(label)}</span>
            `).join("")}
          </div>
          <div class="work-cache-count"><b id="workCacheCount-${escapeHtml(target.id)}">0</b><span>个文件已加入缓存</span></div>
          <div class="work-cache-selected" id="workCacheSelected-${escapeHtml(target.id)}"></div>
          <p class="summary-text">这里的“缓存”用于演示组织本轮复现上下文：用户可以选择数据表、配置、脚本和提示词，再围绕这些文件继续与大模型对话。</p>
          <div class="work-cache-source-layer">
            ${renderDataSourceRegistry(target)}
          </div>
        </div>
      </div>
    </section>
  `;
}

function workCacheStoreKey(targetId) {
  return `reproWorkCache:${targetId}`;
}

function readWorkCache(targetId) {
  try {
    const rows = JSON.parse(localStorage.getItem(workCacheStoreKey(targetId)) || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

function saveWorkCache(targetId, rows) {
  localStorage.setItem(workCacheStoreKey(targetId), JSON.stringify(rows));
}

function updateWorkCacheUi(targetId) {
  const layout = document.querySelector(`.work-cache-layout[data-cache-target="${CSS.escape(targetId)}"]`);
  if (!layout) return;
  const selected = [...layout.querySelectorAll(".work-cache-check:checked")].map((input) => ({
    key: input.dataset.cacheKey,
    label: input.dataset.cacheLabel,
    url: input.dataset.cacheUrl,
  }));
  saveWorkCache(targetId, selected);
  const count = document.querySelector(`#workCacheCount-${CSS.escape(targetId)}`);
  const panel = document.querySelector(`#workCacheSelected-${CSS.escape(targetId)}`);
  if (count) count.textContent = String(selected.length);
  if (panel) {
    panel.innerHTML = selected.length
      ? selected.map((item) => `<a href="${escapeHtml(item.url)}" target="_blank" rel="noopener">${escapeHtml(item.label)}</a>`).join("")
      : `<span>尚未选择文件</span>`;
  }
}

function hydrateWorkCache(targetId) {
  const layout = document.querySelector(`.work-cache-layout[data-cache-target="${CSS.escape(targetId)}"]`);
  if (!layout) return;
  const selected = new Set(readWorkCache(targetId).map((item) => item.key));
  layout.querySelectorAll(".work-cache-check").forEach((input) => {
    input.checked = selected.has(input.dataset.cacheKey);
  });
  updateWorkCacheUi(targetId);
}

function renderStageTwoWorkspace(target) {
  const tasks = asArray(target.data_completion?.tasks);
  const validation = target.data_validation || {};
  const openDataCount = Number(validation.empty_files || 0) + Number(validation.missing_files || 0) + Number(validation.optional_empty_files || 0) + Number(validation.optional_missing_files || 0);
  const generatedFiles = asArray(target.materials?.generated_files);
  return `
    <section class="band stage-two-intro" id="stageTwoWorkspace">
      <div class="band-title">
        <div>
          <h3>阶段二：个性化生成式工作台</h3>
          <small>通过用户与大模型的多轮对话推进下载数据、编写脚本、图表展示和差距解释</small>
        </div>
      </div>
      ${tasks.length ? `
        <div class="stage-two-task-strip">
          ${tasks.slice(0, 6).map((task) => `<span><b>${escapeHtml(task.file)}</b>${escapeHtml(task.action)}</span>`).join("")}
        </div>
      ` : `<p class="summary-text">当前必需数据表已经可运行；仍有 ${escapeHtml(openDataCount)} 个可选或待确认材料可通过对话生成候选文件、修复脚本或展示说明。</p>`}
      ${generatedFiles.length ? `
        <div class="dialogue-output-strip">
          <b>已有对话生成产物</b>
          <span>${generatedFiles.slice(0, 5).map((file) => escapeHtml(file.label || file.name || file.relative)).join(" / ")}</span>
        </div>
      ` : ""}
    </section>
    ${renderPromptChoiceCards(target)}
    ${renderReproChatWindow(target)}
    ${renderWorkFileCache(target)}
  `;
}

function ensurePromptModal() {
  let modal = document.querySelector("#interactionPromptModal");
  if (modal) return modal;
  modal = document.createElement("div");
  modal.id = "interactionPromptModal";
  modal.className = "file-preview-modal hidden";
  modal.innerHTML = `
    <div class="file-preview-backdrop" data-prompt-close="true"></div>
    <section class="file-preview-panel prompt-panel" role="dialog" aria-modal="true" aria-label="下一阶段交互提示词">
      <header>
        <div>
          <h3 id="interactionPromptTitle">下一阶段交互提示词</h3>
          <small>把这段内容交给大模型或作为人工检索任务说明，用于继续补齐数据。</small>
        </div>
        <div class="file-preview-actions">
          <button type="button" id="copyInteractionPrompt">复制提示词</button>
          <button type="button" data-prompt-close="true">关闭</button>
        </div>
      </header>
      <div class="file-preview-body">
        <textarea id="interactionPromptText" spellcheck="false"></textarea>
        <p class="summary-text" id="interactionPromptState"></p>
      </div>
    </section>
  `;
  document.body.appendChild(modal);
  modal.addEventListener("click", async (event) => {
    if (event.target.closest("[data-prompt-close]")) {
      modal.classList.add("hidden");
      return;
    }
    if (event.target.closest("#copyInteractionPrompt")) {
      const textarea = modal.querySelector("#interactionPromptText");
      const state = modal.querySelector("#interactionPromptState");
      textarea.select();
      try {
        await navigator.clipboard.writeText(textarea.value);
        state.textContent = "已复制，可以直接粘贴给大模型继续追问。";
      } catch {
        document.execCommand("copy");
        state.textContent = "已选中并尝试复制；如果剪贴板受限，可以手动复制。";
      }
    }
  });
  return modal;
}

function openInteractionPrompt(prompt, title = "下一阶段交互提示词") {
  const modal = ensurePromptModal();
  modal.querySelector("#interactionPromptTitle").textContent = title;
  modal.querySelector("#interactionPromptText").value = prompt || "";
  modal.querySelector("#interactionPromptState").textContent = "";
  modal.classList.remove("hidden");
}

function renderReproChatMessages(targetId) {
  const history = reproChatHistories[targetId] || [];
  if (!history.length) {
    return `<p class="summary-text">可以直接问：这篇论文的数据该从哪里补？请为模型接口脚本生成第一版数据读取函数；或者根据当前空表设计下一轮检索关键词。</p>`;
  }
  return history.map((item) => `
    <article class="repro-chat-message ${item.role === "assistant" ? "assistant" : "user"}">
      <b>${item.role === "assistant" ? "大模型" : "我"}</b>
      <div>${renderMarkdown(item.content || "")}</div>
      ${renderReproChatArtifacts(item.artifacts)}
    </article>
  `).join("");
}

function renderReproChatArtifacts(artifacts) {
  const items = asArray(artifacts).filter((item) => item?.url);
  if (!items.length) return "";
  return `
    <div class="repro-chat-artifacts">
      <span>本轮生成文件</span>
      <div>
        ${items.map((item) => `
          <a class="repro-chat-artifact-card" href="${escapeHtml(item.url)}" data-file-label="${escapeHtml(item.label || item.relative || "生成文件")}">
            <b>${escapeHtml(item.label || item.relative || "生成文件")}</b>
            <small>${escapeHtml(item.relative || "")}</small>
          </a>
        `).join("")}
      </div>
    </div>
  `;
}

function renderReproChatWindow(target) {
  const targetId = target.id || "";
  const openDataCount = Number(target.data_validation?.empty_files || 0) + Number(target.data_validation?.missing_files || 0) + Number(target.data_validation?.optional_empty_files || 0) + Number(target.data_validation?.optional_missing_files || 0);
  return `
    <section class="band repro-chat-band" data-repro-chat-target="${escapeHtml(targetId)}">
      <div class="band-title">
        <div>
          <h3>大模型多轮协作对话</h3>
          <small>把当前论文的数据缺口、模型结构、代码草稿和缓存文件一起作为上下文，继续补数据、写脚本或调图表</small>
        </div>
      </div>
      <div class="repro-chat-layout">
        <div class="repro-chat-thread" id="reproChatThread-${escapeHtml(targetId)}">
          ${renderReproChatMessages(targetId)}
        </div>
        <form class="repro-chat-form">
          <label>
            <span>需求类型</span>
            <select name="mode">
              <option value="data">补齐数据 / 找来源</option>
              <option value="code">生成或修改复现代码</option>
              <option value="feature">调整工具链或界面功能</option>
              <option value="gap">解释复现差距</option>
              <option value="general">综合协作</option>
            </select>
          </label>
          <label>
            <span>你的需求</span>
            <textarea name="message" rows="5" placeholder="例如：请基于当前可选空表和已有图表，生成数据补全候选文件、修复展示问题，并说明这些产物如何用于汇报。"></textarea>
          </label>
          <div class="repro-chat-actions">
            <button class="primary-button" type="submit">发送给大模型</button>
            <button class="ghost-button repro-chat-demo" type="button" data-demo-message="请基于当前 ${escapeHtml(openDataCount)} 个待补齐或待确认数据表，生成一轮面向本论文的数据补全、功能修复和效果展示方案。">运行示例对话</button>
            <button class="ghost-button repro-chat-clear" type="button">清空对话</button>
          </div>
          <p class="job-state repro-chat-state"></p>
        </form>
      </div>
    </section>
  `;
}

function refreshReproChatThread(targetId) {
  const thread = document.querySelector(`#reproChatThread-${CSS.escape(targetId)}`);
  if (thread) {
    thread.innerHTML = renderReproChatMessages(targetId);
    thread.scrollTop = thread.scrollHeight;
  }
}

async function submitReproChat(form, options = {}) {
  const band = form.closest(".repro-chat-band");
  const targetId = band?.dataset.reproChatTarget || "";
  const messageEl = form.querySelector("textarea[name='message']");
  const stateEl = form.querySelector(".repro-chat-state");
  const button = form.querySelector("button[type='submit']");
  const message = String(options.message || messageEl?.value || "").trim();
  const mode = options.mode || form.querySelector("select[name='mode']")?.value || "general";
  if (!targetId || !message) {
    if (stateEl) stateEl.textContent = "请先输入想让大模型协助的问题。";
    return;
  }
  reproChatHistories[targetId] = reproChatHistories[targetId] || [];
  reproChatHistories[targetId].push({role: "user", content: message});
  refreshReproChatThread(targetId);
  if (!options.keepMessage && messageEl) messageEl.value = "";
  button.disabled = true;
  const demoButton = form.querySelector(".repro-chat-demo");
  if (demoButton) demoButton.disabled = true;
  if (stateEl) stateEl.textContent = options.demo ? "正在运行定制化示例..." : "正在调用大模型接口...";
  try {
    const response = await fetch("/api/repro-chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        target_id: targetId,
        mode,
        message,
        history: reproChatHistories[targetId].slice(-8),
        demo: Boolean(options.demo)
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "大模型调用失败");
    reproChatHistories[targetId].push({role: "assistant", content: payload.answer || "", artifacts: payload.artifacts || []});
    if (stateEl) stateEl.textContent = "已返回，可继续追问。";
  } catch (error) {
    reproChatHistories[targetId].push({role: "assistant", content: `调用失败：${error.message || String(error)}\n\n请检查大模型接口配置，或稍后重试。`});
    if (stateEl) stateEl.textContent = "大模型调用失败，错误已显示在对话中。";
  } finally {
    button.disabled = false;
    if (demoButton) demoButton.disabled = false;
    refreshReproChatThread(targetId);
  }
}

function renderDataCompletionAssistant(target) {
  const completion = target.data_completion || {};
  const tasks = asArray(completion.tasks);
  if (!tasks.length) return "";
  return `
    <section class="band data-completion-band">
      <div class="band-title">
        <div>
          <h3>下一阶段：数据补齐交互入口</h3>
          <small>把“空数据表”转成可继续问大模型、人工检索和回填 CSV 的任务包</small>
        </div>
        <button class="ghost-button prompt-open-button" type="button" data-prompt-kind="data-completion">打开补齐提示词</button>
      </div>
      <div class="data-completion-layout">
        <article class="data-completion-guide">
          <h4>推荐流程</h4>
          ${listBlock(completion.instructions, 4)}
          <div class="repro-stage-actions">
            ${target.links?.source_trace_md ? `<a href="${escapeHtml(target.links.source_trace_md)}" target="_blank" rel="noopener">来源追踪</a>` : ""}
            ${target.links?.model_spec_md ? `<a href="${escapeHtml(target.links.model_spec_md)}" target="_blank" rel="noopener">模型规范</a>` : ""}
          </div>
        </article>
        <div class="data-task-grid">
          ${tasks.slice(0, 14).map((task) => `
            <article class="data-task-card">
              <div>
                <b>${escapeHtml(task.file)}</b>
                <span>${escapeHtml(task.status)}</span>
              </div>
              <p>${escapeHtml(task.action)}</p>
              <small>${escapeHtml(task.source_hint)}</small>
              ${task.url ? `<a href="${escapeHtml(task.url)}" data-file-label="${escapeHtml(task.file)}">查看 CSV 模板</a>` : ""}
            </article>
          `).join("")}
        </div>
      </div>
      <textarea class="hidden" id="dataCompletionPrompt">${escapeHtml(completion.prompt || "")}</textarea>
    </section>
  `;
}

function renderDialogueStructure(target) {
  const report = asArray(target.reports).find((item) => item.title === "大模型多轮对话结构");
  const rounds = [
    ["1", "证据片段进入提示词", "工具从论文文本中筛选数据表、模型公式、实验设置和结果对齐证据，作为提示词的证据区。"],
    ["2", "审计提示词与结构", "把论文元数据和证据填入复现审计模板，并要求大模型按固定结构返回阻塞项。"],
    ["3", "模型提示词与结构", "把审计结果和证据继续送入模型规范模板，抽取集合、参数、变量、目标函数和约束。"],
    ["4", "结构化结果生成代码", "脚手架读取 audit/model_spec，生成 data/configs/src/reports，目标论文代码卡片展示这些生成结果。"],
    ["5", "结果回看再追问", "可根据空数据、缺失参数和复现差距继续组织下一轮提示词，让大模型补代码或补分析。"]
  ];
  const promptAssets = asArray(target.llm_prompt_assets);
  return `
    <div class="dialogue-flow">
      ${rounds.map(([index, title, text]) => `
        <article>
          <b>${escapeHtml(index)}</b>
          <div>
            <h4>${escapeHtml(title)}</h4>
            <p>${escapeHtml(text)}</p>
          </div>
        </article>
      `).join("")}
    </div>
    <div class="prompt-asset-grid">
      ${promptAssets.map((asset) => `
        <a class="prompt-asset-card" href="${escapeHtml(asset.url)}" data-file-label="${escapeHtml(asset.label || asset.name)}">
          <span>${escapeHtml(asset.kind || "file")}</span>
          <b>${escapeHtml(asset.label || asset.name)}</b>
          <small>${escapeHtml(asset.relative)}</small>
        </a>
      `).join("")}
    </div>
    ${report?.text ? `<div class="repro-report-excerpt">${renderMarkdown(report.text)}</div>` : ""}
  `;
}

function renderCodeArtifacts(target) {
  const files = asArray(target.code_artifacts);
  const validationReport = asArray(target.reports).find((item) => item.title === "大模型代码生成验证" && item.text);
  if (!files.length && !validationReport) return `<p class="summary-text">暂无代码产物。</p>`;
  const groups = [
    ["generated", "大模型引导生成的目标论文代码", files.filter((file) => file.group === "generated")],
    ["toolchain", "复现工具链源码", files.filter((file) => file.group !== "generated")]
  ];
  return `
    ${validationReport ? `<div class="repro-report-excerpt code-validation-report">${renderMarkdown(validationReport.text)}</div>` : ""}
    ${groups.map(([, title, groupFiles]) => groupFiles.length ? `
      <div class="code-artifact-section">
        <div class="module-head"><h4>${escapeHtml(title)}</h4><small>${groupFiles.length} 个</small></div>
        <div class="code-artifact-grid">
          ${groupFiles.slice(0, 10).map((file) => `
            <a class="code-artifact-card" href="${escapeHtml(file.url)}" data-file-label="${escapeHtml(file.name)}">
              <span>${escapeHtml(file.name)}</span>
              <b>${escapeHtml(file.role || "代码文件")}</b>
              <small>${escapeHtml(file.relative || "")}</small>
              <em>${escapeHtml(compactText(file.preview || "打开查看完整代码", 180))}</em>
            </a>
          `).join("")}
        </div>
      </div>
    ` : "").join("")}
  `;
}

function renderReproFigures(target) {
  const seen = new Set();
  const figures = asArray(target.figures)
    .filter((figure) => {
      const relative = String(figure.relative || figure.url || "");
      const key = relative.replace(/\.(png|jpg|jpeg|svg)$/i, "");
      if (seen.has(key)) return false;
      seen.add(key);
      return !/\.svg$/i.test(relative) || !asArray(target.figures).some((item) => String(item.relative || item.url || "").replace(/\.(png|jpg|jpeg|svg)$/i, "") === key && /\.(png|jpg|jpeg)$/i.test(String(item.relative || item.url || "")));
    })
    .slice(0, 4);
  if (!figures.length) return `<p class="summary-text">暂无图形产物。</p>`;
  return `
    <div class="repro-figure-grid">
      ${figures.map((figure) => `
        <figure>
          <a href="${escapeHtml(figure.url)}" target="_blank" rel="noopener">
            <img src="${escapeHtml(figure.url)}" alt="${escapeHtml(figure.label)}" loading="lazy">
          </a>
          <figcaption>${escapeHtml(figure.label)}</figcaption>
        </figure>
      `).join("")}
    </div>
  `;
}

function renderTargetDashboard(target) {
  const validation = target.data_validation || {};
  const progress = targetProgress(target);
  const dataTasks = asArray(target.data_completion?.tasks).length || (Number(validation.empty_files || 0) + Number(validation.missing_files || 0));
  const promptCount = asArray(target.llm_prompt_assets).length + (target.data_completion?.prompt ? 1 : 0);
  return `
    <section class="band repro-target" id="repro-${escapeHtml(target.id)}">
      <div class="band-title">
        <div>
          <h3>${escapeHtml(target.title)}</h3>
          <small>${escapeHtml([asArray(target.authors).join(", "), target.year, target.venue].filter(Boolean).join(" · "))}</small>
        </div>
        <small>${escapeHtml(target.id)}</small>
      </div>
      <div class="repro-status-strip">
        <span><b>论文拆解</b>${escapeHtml(progress.completed)}/${escapeHtml(progress.steps.length)}</span>
        <span><b>数据表</b>${escapeHtml(valueOrDash(validation.complete_files))} 个完整</span>
        <span><b>待确认</b>${escapeHtml(dataTasks)} 项</span>
        <span><b>图表</b>${escapeHtml(asArray(target.figures).length)} 个</span>
        <span class="role"><b>角色</b>${escapeHtml(valueOrDash(target.role || "复现目标"))}</span>
      </div>
      <div class="repro-link-row">
        ${Object.entries(target.links || {}).filter(([, url]) => url).map(([key, url]) => `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(key.replace(/_/g, " "))}</a>`).join("")}
      </div>
    </section>

    ${renderStageVerdict(target)}

    ${renderReproductionProgress(target)}

    ${renderStageOneExtraction(target)}

    ${renderStageTwoWorkspace(target)}

    <section class="band" id="reproFigures">
      <div class="band-title"><h3>结果展示</h3><small>仅保留关键图表，完整 CSV 与 SVG 可在阶段二文件缓存中打开</small></div>
      ${renderReproFigures(target)}
    </section>

    <details class="repro-details repro-optional-details">
      <summary>展开查看完整材料索引、代码和差距报告</summary>
      <section class="band">
        <div class="band-title"><h3>代码与工具链</h3><small>生成脚本和底层工具源码</small></div>
        ${renderCodeArtifacts(target)}
      </section>
      <section class="band">
        <div class="band-title"><h3>差距报告</h3><small>结果对齐、复现限制与后续工作</small></div>
        ${renderCheckRows(target.result_alignment, "target", "evidence", 6)}
        ${asArray(target.reports).filter((item) => item.text).slice(0, 3).map((item) => `
          <details class="repro-details">
            <summary>${escapeHtml(item.title)}${item.relative ? ` · ${escapeHtml(item.relative)}` : ""}</summary>
            <div class="repro-report-excerpt">${renderMarkdown(item.text)}</div>
          </details>
        `).join("")}
      </section>
    </details>
  `;
}

function targetHash(target) {
  return `#repro-${encodeURIComponent(target.id)}`;
}

function targetFromHash() {
  const raw = decodeURIComponent(String(window.location.hash || "").replace(/^#repro-/, ""));
  if (!raw || raw === window.location.hash) return null;
  return asArray(reproductionData.targets).find((target) => String(target.id) === raw) || null;
}

function targetStatusSummary(target) {
  const validation = target.data_validation || {};
  const figures = asArray(target.figures).length;
  const codeFiles = asArray(target.code_artifacts).filter((item) => String(item.relative || "").startsWith("src/")).length;
  const promptCount = asArray(target.llm_prompt_assets).length + (target.data_completion?.prompt ? 1 : 0);
  const dataText = validation.complete_files !== undefined && validation.complete_files !== null ? `${validation.complete_files} 表` : "-";
  return {dataText, figures, codeFiles, promptCount};
}

function renderReproductionCard(target) {
  const summary = targetStatusSummary(target);
  const firstFigure = asArray(target.figures)[0];
  const tags = [
    target.role || target.recommended_role,
    target.year,
    `${summary.figures} 图`,
    `${summary.codeFiles} 代码`,
  ].filter(Boolean);
  return `
    <article class="repro-paper-card">
      <a class="repro-card-main" href="?view=reproduction${targetHash(target)}" data-repro-target="${escapeHtml(target.id)}">
        ${firstFigure?.url ? `
          <div class="repro-card-visual">
            <img src="${escapeHtml(firstFigure.url)}" alt="${escapeHtml(firstFigure.label || target.title)}" loading="lazy">
          </div>
        ` : `
          <div class="repro-card-visual repro-card-placeholder">
            <span>${escapeHtml((target.title || target.id).slice(0, 2).toUpperCase())}</span>
          </div>
        `}
        <div class="repro-card-body">
          <div>
            <h3>${escapeHtml(compactText(target.title || target.id, 118))}</h3>
            <p>${escapeHtml(compactText([asArray(target.authors).join(", "), target.venue].filter(Boolean).join(" · "), 110))}</p>
          </div>
          <div class="repro-card-metrics">
            <span><b>${escapeHtml(summary.dataText)}</b><small>数据完整</small></span>
            <span><b>${escapeHtml(summary.figures)}</b><small>展示图</small></span>
            <span><b>${escapeHtml(summary.promptCount)}</b><small>提示词</small></span>
          </div>
          <div class="repro-card-tags">
            ${tags.slice(0, 5).map((tag) => `<em>${escapeHtml(tag)}</em>`).join("")}
          </div>
        </div>
      </a>
    </article>
  `;
}

function renderReproductionPool(targets) {
  return `
    <section class="metrics repro-overview-metrics">
      ${metric("复现目标", `${targets.length} 个`)}
      ${metric("代码模块", `${targets.reduce((total, target) => total + asArray(target.code_artifacts).length, 0)} 个`)}
      ${metric("图表产物", `${targets.reduce((total, target) => total + asArray(target.figures).length, 0)} 个`)}
      ${metric("仓库根目录", reproductionData.repo_root || "-")}
    </section>

    <section class="band repro-pool-band">
      <div class="band-title">
        <div>
          <h3>复现论文卡片池</h3>
          <small>点击卡片进入该论文的审计、数据、模型、代码、图表和差距详情</small>
        </div>
        <small>${targets.length} 个目标</small>
      </div>
      <div class="repro-card-pool">
        ${targets.length ? targets.map(renderReproductionCard).join("") : `<p class="summary-text">还没有发现 runs/&lt;target&gt;/target.yaml。</p>`}
      </div>
    </section>

    <section class="band">
      <div class="band-title"><h3>通用复现流程</h3><small>repro_cli 可复用入口</small></div>
      <div class="command-strip compact-command-strip">
        ${asArray(reproductionData.toolchain_commands).map((command) => `<code>${escapeHtml(command)}</code>`).join("")}
      </div>
    </section>
  `;
}

function renderReproductionDetail(target) {
  return `
    <section class="repro-detail-toolbar">
      <a class="ghost-button" href="?view=reproduction" id="reproBackToPool">返回卡片池</a>
      <a class="ghost-button" href="?view=reproduction${targetHash(target)}">单篇链接</a>
    </section>
    ${renderTargetDashboard(target)}
  `;
}

function renderReproductionDashboard() {
  setWorkbenchMode("reproduction");
  currentDirection = null;
  currentPaper = null;
  const targets = asArray(reproductionData.targets);
  const selectedTarget = targetFromHash();
  if (selectedTarget) {
    setHeader("reproduction", "论文复现详情", "阶段一明确数据、模型和环境；阶段二通过多轮对话推进下载数据、写脚本和看图表。");
    canvas.innerHTML = renderReproductionDetail(selectedTarget);
    hydrateWorkCache(selectedTarget.id || "");
  } else {
    setHeader("reproduction", "论文复现工具链", "先选择论文，再进入两阶段复现工作台。");
    canvas.innerHTML = renderReproductionPool(targets);
  }
  typesetMath();
}

startButton?.addEventListener("click", async () => {
  if (activePollTimer) clearTimeout(activePollTimer);
  const clientJobId = createClientJobId();
  visibleJobId = clientJobId;
  rememberJob({id: clientJobId, status: "queued"});
  startButton.disabled = true;
  progressPanel?.classList.remove("hidden");
  state.textContent = "正在提交任务...";
  const payload = {
    client_job_id: clientJobId,
    topic: document.querySelector("#topic").value,
    mode: document.querySelector("#mode").value,
    pdf_dir: document.querySelector("#pdfDir")?.value || "input_pdfs",
    all_papers: document.querySelector("#allPapers")?.checked || false,
    max_papers: Number(document.querySelector("#maxPapers").value || 0) || null,
    max_results: Number(document.querySelector("#maxResults").value || 0) || null,
    year_from: Number(document.querySelector("#yearFrom")?.value || 0) || null,
    year_to: Number(document.querySelector("#yearTo")?.value || 0) || null,
    filter_and: "",
    run_parts: document.querySelector("#runParts").value,
    topic_clauses: collectClauses()
  };
  let response;
  let job;
  try {
    response = await fetch("/api/jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      keepalive: true,
      body: JSON.stringify(payload)
    });
    job = await response.json();
  } catch {
    state.textContent = "任务已交给后台启动，页面可继续切换查看。";
    schedulePoll(clientJobId);
    return;
  }
  if (!response.ok) {
    state.textContent = job.error || "提交失败";
    startButton.disabled = false;
    return;
  }
  visibleJobId = job.id;
  if (job.id !== clientJobId) {
    updateTrackedJob({id: clientJobId, status: "completed", progress: {run_id: ""}});
  }
  rememberJob(job);
  renderProgress(job);
  state.textContent = "任务已启动，后台运行中...";
  pollJob(job.id);
});

addTopicClause?.addEventListener("click", () => addClause());
document.querySelector("#mode")?.addEventListener("change", syncModeControls);
literatureViewButton?.addEventListener("click", () => {
  setWorkbenchMode("literature");
  history.replaceState(null, "", overviewUrl());
  renderOverview();
});
reproductionViewButton?.addEventListener("click", () => {
  history.replaceState(null, "", "?view=reproduction");
  renderReproductionDashboard();
});
window.addEventListener("hashchange", () => {
  const params = new URLSearchParams(window.location.search);
  if (params.get("view") === "reproduction") {
    renderReproductionDashboard();
  }
});
document.addEventListener("click", (event) => {
  const cacheButton = event.target.closest("[data-cache-action]");
  if (cacheButton) {
    event.preventDefault();
    const layout = cacheButton.closest(".band")?.querySelector(".work-cache-layout");
    const targetId = layout?.dataset.cacheTarget || "";
    if (!targetId) return;
    const action = cacheButton.dataset.cacheAction;
    layout.querySelectorAll(".work-cache-check").forEach((input) => {
      if (action === "clear") input.checked = false;
      if (action === "select-data") input.checked = String(input.dataset.cacheKey || "").includes(":data:");
    });
    updateWorkCacheUi(targetId);
    return;
  }
  const promptButton = event.target.closest(".prompt-open-button");
  if (promptButton) {
    event.preventDefault();
    const prompt = document.querySelector("#dataCompletionPrompt")?.value || "";
    openInteractionPrompt(prompt, "数据补齐交互提示词");
    return;
  }
  const link = event.target.closest(".repro-material-card, .prompt-asset-card, .code-artifact-card, .data-task-card a, .repro-chat-artifact-card");
  if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  openFilePreview(link.href, link.dataset.fileLabel || link.querySelector("b")?.textContent || "文件预览");
});
document.addEventListener("change", (event) => {
  const input = event.target.closest(".work-cache-check");
  if (!input) return;
  const layout = input.closest(".work-cache-layout");
  const targetId = layout?.dataset.cacheTarget || "";
  if (targetId) updateWorkCacheUi(targetId);
});
document.addEventListener("submit", (event) => {
  const form = event.target.closest(".repro-chat-form");
  if (!form) return;
  event.preventDefault();
  submitReproChat(form);
});
document.addEventListener("click", (event) => {
  const demoButton = event.target.closest(".repro-chat-demo");
  if (demoButton) {
    event.preventDefault();
    const form = demoButton.closest(".repro-chat-form");
    if (!form) return;
    form.querySelector("select[name='mode']").value = "data";
    submitReproChat(form, {
      demo: true,
      mode: "data",
      message: demoButton.dataset.demoMessage || "请给出一个定制化复现推进方案。"
    });
    return;
  }
  const clearButton = event.target.closest(".repro-chat-clear");
  if (!clearButton) return;
  const band = clearButton.closest(".repro-chat-band");
  const targetId = band?.dataset.reproChatTarget || "";
  if (!targetId) return;
  reproChatHistories[targetId] = [];
  refreshReproChatThread(targetId);
  const stateEl = band.querySelector(".repro-chat-state");
  if (stateEl) stateEl.textContent = "已清空本页对话记录。";
});
syncModeControls();
renderQueryMeta();
resumeTrackedJobForCurrentRun();

if (urlParams.get("view") === "reproduction") {
  renderReproductionDashboard();
} else if (initialView.layer === "paper") {
  setWorkbenchMode("literature");
  renderPaper(initialView.direction_id, initialView.paper_id);
} else if (initialView.layer === "direction") {
  setWorkbenchMode("literature");
  renderDirection(initialView.direction_id);
} else {
  setWorkbenchMode("literature");
  renderOverview();
}
