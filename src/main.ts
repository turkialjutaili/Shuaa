import './style.css';

type View = 'overview' | 'experiments' | 'compare' | 'protocol' | 'submit';
type Phase = 'validation' | 'test';
type Participant = { id: string; name: string };
type TrainingComponent = {
  run_id: string; model_name: string; seed: number; epochs_completed: number; configured_epochs: number;
  optimizer: string; initial_learning_rate: number; weighted_loss: boolean;
};
type Training = { ensemble_method: 'single_model' | 'probability_mean'; components: TrainingComponent[] };
type Result = {
  id: string; evaluation_id?: string; participant: string; model_name: string; phase: Phase; status: 'success' | 'failed';
  accuracy?: number; macro_f1?: number; balanced_accuracy?: number;
  per_class?: { label: string; precision: number; recall: number; f1: number; support: number }[];
  confusion_matrix?: number[][]; evaluated_at: string; source_commit?: string; paper_url?: string;
  code_url?: string; checkpoint_sha256?: string; split_hash?: string; training?: Training; error?: string;
};
type Board = { schema_version: number; updated_at: string | null; classes: string[]; participants: Participant[]; results: Result[] };
const repo = 'https://github.com/turkialjutaili/Shuaa';
const defaultParticipants = [{ id: 'turki', name: 'Turki' }, { id: 'muhannad', name: 'Muhannad' }, { id: 'mazen', name: 'Mazen' }];
let board: Board = { schema_version: 1, updated_at: null, classes: [], participants: defaultParticipants, results: [] };
let lang = (() => { try { return localStorage.getItem('shuaa-language') === 'ar' ? 'ar' : 'en'; } catch { return 'en'; } })();
let view: View = 'overview';
let phase: Phase = 'validation';
let participantFilter = 'all';
let search = '';
let loading = true;
let loadError = false;
let selected: string[] = [];
let modalId: string | null = null;
let menuOpen = false;
const app = document.querySelector<HTMLDivElement>('#app')!;
const t = (en: string, ar: string) => lang === 'ar' ? ar : en;
const escape = (s: unknown) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
const percent = (n?: number) => n === undefined || !Number.isFinite(n) ? '—' : `${(n * 100).toFixed(2)}%`;
const learningRate = (n: number) => Number.isFinite(n) ? n.toFixed(8).replace(/0+$/, '').replace(/\.$/, '') : '—';
const name = (id: string) => ({ turki: t('Turki', 'تركي'), muhannad: t('Muhannad', 'مهند'), mazen: t('Mazen', 'مازن') }[id] || board.participants.find(p => p.id === id)?.name || id);
const date = (value?: string | null) => value && !Number.isNaN(Date.parse(value)) ? new Intl.DateTimeFormat(lang === 'ar' ? 'ar-SA' : 'en-GB', { day: 'numeric', month: 'short', year: 'numeric', calendar: 'gregory' }).format(new Date(value)) : t('Not evaluated yet', 'لم يجرِ التقييم بعد');
const icons: Record<string, string> = {
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  compare: '<path d="M7 3v18m10-18v18M3 7h8m2 10h8"/><circle cx="7" cy="7" r="2"/><circle cx="17" cy="17" r="2"/>',
  book: '<path d="M12 6C8 3 4 4 3 5v15c3-2 6-1 9 1 3-2 6-3 9-1V5c-1-1-5-2-9 1Zm0 0v15"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
  arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
  external: '<path d="M14 3h7v7m0-7L10 14M10 3H4a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1v-6"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  layers: '<path d="m12 3 10 5-10 5L2 8Zm-9 10 9 5 9-5M3 18l9 5 9-5"/>',
  trophy: '<path d="M8 3h8v7a4 4 0 0 1-8 0Zm0 2H4v3a4 4 0 0 0 4 4m8-7h4v3a4 4 0 0 1-4 4m-4 2v6m-4 1h8"/>',
  search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>',
  github: '<path d="M9 19c-4 1-4-2-6-2m12 5v-4c0-1-.4-2-1-2 3-.4 6-1.5 6-6 0-1-.4-2-1-3 0-1 0-2-.3-3-2 0-3 1-3 1a12 12 0 0 0-7 0S7 4 5 4c-.3 1-.3 2 0 3-.7 1-1 2-1 3 0 4.5 3 5.6 6 6-.6.5-1 1-1 2v4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  globe: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
};
const icon = (key: string, cls = '') => `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[key] || icons.sun}</svg>`;
const link = (url: string | undefined, label: string) => {
  if (!url) return '';
  try { if (!['https:', 'http:'].includes(new URL(url).protocol)) return ''; } catch { return ''; }
  return `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)} ${icon('external')}</a>`;
};
const resultKey = (r: Result) => r.evaluation_id || `${r.id}:${r.phase}:${r.evaluated_at}`;
const good = () => board.results.filter(r => r.status === 'success' && r.phase === phase && Number.isFinite(r.accuracy) && Number.isFinite(r.macro_f1));
const sorted = () => good().sort((a, b) => (b.accuracy! - a.accuracy!) || (b.macro_f1! - a.macro_f1!));
const avatar = (id: string) => `<span class="avatar ${escape(id)}">${escape(name(id).slice(0, 1))}</span>`;
const phaseSwitch = () => `<div class="segmented" aria-label="${t('Evaluation phase', 'مرحلة التقييم')}"><button data-phase="validation" class="${phase === 'validation' ? 'active' : ''}" aria-pressed="${phase === 'validation'}">${t('Validation', 'التحقق')}</button><button data-phase="test" class="${phase === 'test' ? 'active' : ''}" aria-pressed="${phase === 'test'}">${t('Final test', 'الاختبار النهائي')}</button></div>`;
const empty = (title: string, body: string, action = true) => `<div class="empty-state"><span class="empty-symbol">${icon('activity')}</span><h3>${title}</h3><p>${body}</p>${action ? `<button class="text-button" data-view="submit">${t('Submit the first model', 'أضف النموذج الأول')} ${icon('arrow')}</button>` : ''}</div>`;
const titles = () => ({ overview: t('Overview', 'نظرة عامة'), experiments: t('Experiments', 'التجارب'), compare: t('Compare models', 'مقارنة النماذج'), protocol: t('Research & protocol', 'البحث والمنهجية'), submit: t('Submit a model', 'إضافة نموذج') });

function shell() {
  const nav = [['overview', 'grid'], ['experiments', 'activity'], ['compare', 'compare'], ['protocol', 'book']] as const;
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  document.title = `${titles()[view]} · ${t('Shuaa', 'شعاع')}`;
  app.innerHTML = `<div class="layout"><aside id="site-navigation" class="sidebar ${menuOpen ? 'open' : ''}"><button class="sidebar-close" data-action="menu" aria-label="${t('Close navigation', 'إغلاق التنقل')}">${icon('close')}</button><button class="brand" data-view="overview" aria-label="${t('Shuaa home', 'الرئيسية شعاع')}"><span class="brand-mark">${icon('sun')}</span><span>${t('SHUAA', 'شعاع')}<small>${t('SOLAR INTELLIGENCE', 'ذكاء الطاقة الشمسية')}</small></span></button>
    <div class="workspace"><span class="workspace-dot"></span><div>${t('Research workspace', 'مساحة البحث')}<small>${t('Solar anomaly benchmark', 'معيار شذوذ الألواح الشمسية')}</small></div><span class="workspace-tag">V1</span></div>
    <p class="nav-label">${t('WORKSPACE', 'مساحة العمل')}</p><nav>${nav.map(([v, i]) => `<button class="nav-item ${view === v ? 'active' : ''}" data-view="${v}" ${view === v ? 'aria-current="page"' : ''}>${icon(i)}<span>${titles()[v]}</span>${v === 'experiments' ? `<span class="nav-count">${board.results.length}</span>` : ''}</button>`).join('')}</nav>
    <div class="sidebar-team"><p class="nav-label">${t('THE RESEARCH TEAM', 'فريق البحث')}</p>${board.participants.map(p => `<div class="team-person">${avatar(p.id)}<div>${escape(name(p.id))}<small>${t('Researcher', 'باحث')}</small></div><span class="team-dot"></span></div>`).join('')}</div>
    <div class="sidebar-bottom"><div class="open-science">${icon('book')}<span>${t('Built on open science.', 'مبني على العلم المفتوح.')}<small>${t('Reproducible by design.', 'قابل لإعادة التجربة.')}</small></span></div><a class="repo-link" href="${repo}" target="_blank" rel="noopener noreferrer">${icon('github')}${t('Project repository', 'مستودع المشروع')}${icon('external')}</a></div></aside>
    ${menuOpen ? `<button class="sidebar-overlay" data-action="menu" aria-label="${t('Close navigation', 'إغلاق التنقل')}"></button>` : ''}
    <div class="main"><header class="topbar"><div class="breadcrumb"><button class="icon-button mobile-menu" data-action="menu" aria-controls="site-navigation" aria-expanded="${menuOpen}" aria-label="${t('Toggle navigation', 'فتح التنقل')}">${icon('menu')}</button><span class="breadcrumb-brand">${t('Workspace', 'مساحة العمل')}</span><span class="slash">/</span><strong>${titles()[view]}</strong></div><div class="top-actions"><span class="public-badge"><span></span>${t('Public benchmark', 'معيار عام')}</span><button class="language-button" data-action="language" aria-label="${t('Switch to Arabic', 'Switch to English')}">${icon('globe')} ${t('العربية', 'English')}</button></div></header>
    <main id="main-content" tabindex="-1"><div class="page-heading"><div><div class="eyebrow">${t('SHUAA RESEARCH LAB', 'مختبر شعاع البحثي')}</div><h1>${titles()[view]}</h1><p>${view === 'overview' ? t('One dataset. Shared standards. Better solar intelligence.', 'بيانات موحدة. معايير مشتركة. ذكاء أفضل للطاقة الشمسية.') : view === 'experiments' ? t('Every model, every result, a reproducible record.', 'كل نموذج وكل نتيجة في سجل قابل لإعادة التجربة.') : view === 'compare' ? t('See what changes when the architecture does.', 'قارن النتائج عند تغيير معمارية النموذج.') : view === 'protocol' ? t('A fair comparison starts with a shared method.', 'المقارنة العادلة تبدأ بمنهجية مشتركة.') : t('Your next experiment belongs here.', 'هنا مكان تجربتك القادمة.')}</p></div>${view !== 'submit' ? `<button class="primary-button" data-view="submit">${icon('upload')}${t('Submit a model', 'إضافة نموذج')}</button>` : ''}</div>
    ${loadError ? `<div class="notice error" role="alert">${t('We couldn’t load the evaluation record. The leaderboard below is unavailable.', 'تعذر تحميل سجل التقييم. لوحة الترتيب غير متاحة حاليًا.')} <button data-action="reload">${t('Try again', 'إعادة المحاولة')}</button></div>` : ''}
    ${loading ? `<div class="notice" role="status">${t('Loading the evaluation record…', 'جارٍ تحميل سجل التقييم…')}</div>` : ''}
    ${view === 'overview' ? overview() : view === 'experiments' ? experiments() : view === 'compare' ? compare() : view === 'protocol' ? protocol() : submit()}
    ${view === 'overview' || view === 'experiments' ? reportedExperiments() : ''}
    <footer><span>© ${new Date().getFullYear()} ${t('Shuaa Research', 'شعاع للأبحاث')}</span><span>${t('Open research. Measurable progress.', 'بحث مفتوح. تقدم قابل للقياس.')}</span><a href="${repo}" target="_blank" rel="noopener noreferrer">GitHub ${icon('external')}</a></footer></main></div></div>${modalId ? details(modalId) : ''}`;
  if (modalId) { document.querySelector<HTMLButtonElement>('.modal-close')?.focus(); }
}

// Researcher-reported local runs are displayed separately from evaluator-generated rankings.
function reportedExperiments() {
  const release = `${repo}/releases/tag/mazen-swin-small-local-20260919`;
  const archive = `${repo}/releases/download/mazen-swin-small-local-20260919/mazen_swin_small_results.zip`;
  return `<section class="card recent-card" aria-labelledby="reported-experiments-title">
    <div class="card-heading"><div><h2 id="reported-experiments-title">${t('Experiment results', 'نتائج التجارب')}</h2><p>${t('Researcher-reported results on individual data splits.', 'نتائج يشاركها الباحثون على تقسيماتهم الخاصة للبيانات.')}</p></div><span class="status pending">${t('Reported locally', 'تقييم محلي')}</span></div>
    <div class="table-wrap"><table><thead><tr><th>${t('Researcher / model', 'الباحث / النموذج')}</th><th>${t('Test accuracy', 'دقة الاختبار')}</th><th>Macro-F1</th><th>${t('Test images', 'صور الاختبار')}</th></tr></thead>
    <tbody><tr><td><div class="researcher-cell">${avatar('mazen')}<div><strong>${escape(name('mazen'))}</strong><small>Swin-Small · ImageNet-22K → 1K</small></div></div></td><td class="metric">${percent(0.849799732977303)}</td><td class="metric">${percent(0.7383886177348331)}</td><td>2,996</td></tr></tbody></table></div>
    <div class="padded"><p>${t('34 epochs completed out of 40; best checkpoint: epoch 28. Test predictions average logits from the original image and its horizontal flip (TTA).', 'اكتملت 34 دورة من أصل 40؛ أفضل نسخة من الدورة 28. يستخدم الاختبار متوسط مخرجات الصورة ونسختها المعكوسة أفقيًا (TTA).')}</p>
    <p class="notice">${t('Different split from the shared benchmark. These scores are not part of the verified ranking and are not directly comparable with its scores.', 'تقسيم مختلف عن المعيار المشترك. هذه الأرقام خارج الترتيب الموحّد ولا تُقارن مباشرة بنتائجه.')}</p>
    <div class="detail-links">${link(release, t('Experiment details', 'تفاصيل التجربة'))}${link(archive, t('Download charts and reports', 'تحميل الرسوم والتقارير'))}</div></div>
  </section>`;
}

function overview() {
  const successful = sorted();
  const best = successful[0];
  const ranked = board.participants.map(p => ({ ...p, result: successful.find(r => r.participant === p.id) })).sort((a, b) => a.result && b.result ? (b.result.accuracy! - a.result.accuracy!) || (b.result.macro_f1! - a.result.macro_f1!) : a.result ? -1 : b.result ? 1 : 0);
  let rank = 0;
  return `<section class="hero"><div class="hero-content"><span class="hero-kicker"><span></span>${t('THE SOLAR ANOMALY CHALLENGE', 'تحدي شذوذ الألواح الشمسية')}</span><h2>${t('A clearer picture.<br>A stronger model.', 'رؤية أوضح.<br>نموذج أقوى.')}</h2><p>${t('Turning infrared images into insight. Build, benchmark, and improve solar module classification together.', 'نحوّل الصور الحرارية إلى معرفة. نبني نماذج تصنيف الألواح الشمسية ونقارنها ونطوّرها معًا.')}</p><button class="hero-link" data-view="protocol">${t('Explore the benchmark', 'اكتشف معيار المقارنة')}${icon('arrow')}</button></div><div class="solar-art" aria-hidden="true"><div class="sun-orbit orbit-one"></div><div class="sun-orbit orbit-two"></div><div class="sun-disc"></div><div class="solar-array">${Array.from({ length: 15 }, (_, i) => `<i class="panel-cell c${i}"></i>`).join('')}</div><span class="art-coordinate">24.7136° N / 46.6753° E</span><span class="art-caption">INFRARED → INTELLIGENCE</span></div></section>
    <section class="stats-grid" aria-label="${t('Benchmark summary', 'ملخص المعيار')}">${stat('layers', '20,000', t('Infrared images', 'صورة حرارية'), t('Raptor Maps dataset', 'بيانات Raptor Maps'))}${stat('grid', '12', t('Classification categories', 'فئة تصنيف'), t('Normal + 11 anomaly types', 'سليم + 11 نوع شذوذ'))}${stat('activity', String(good().length), t('Evaluated models', 'نموذج مُقيّم'), t('In the selected phase', 'في المرحلة المحددة'))}${stat('trophy', percent(best?.accuracy), t('Leading accuracy', 'أعلى دقة'), best ? name(best.participant) : t('Waiting for the first result', 'بانتظار أول نتيجة'))}</section>
    <div class="overview-columns"><section class="card leaderboard"><div class="card-heading"><div><h2>${t('The leaderboard', 'لوحة الترتيب')}</h2><p>${t('A friendly competition. A common goal.', 'منافسة ودية. هدف مشترك.')}</p></div>${phaseSwitch()}</div><div class="table-wrap"><table><thead><tr><th class="rank-cell">#</th><th>${t('Researcher / model', 'الباحث / النموذج')}</th><th>${t('Accuracy', 'الدقة')}</th><th>Macro-F1</th><th>${t('Status', 'الحالة')}</th></tr></thead><tbody>${ranked.map((p, i) => {
      const r = p.result;
      if (r && (i === 0 || r.accuracy !== ranked[i - 1].result?.accuracy || r.macro_f1 !== ranked[i - 1].result?.macro_f1)) rank = i + 1;
      return `<tr><td class="rank-cell">${r ? `<span class="rank ${rank === 1 ? 'first' : ''}">${String(rank).padStart(2, '0')}</span>` : '<span class="muted">—</span>'}</td><td><div class="researcher-cell">${avatar(p.id)}<div><strong>${escape(name(p.id))}</strong>${r ? `<button class="model-link" data-details="${escape(resultKey(r))}">${escape(r.model_name)}</button>` : `<small>${t('No evaluated model yet', 'لا يوجد نموذج مُقيّم بعد')}</small>`}</div></div></td><td class="metric">${percent(r?.accuracy)}</td><td class="metric muted">${percent(r?.macro_f1)}</td><td><span class="status ${r ? 'success' : 'pending'}">${r ? t('Evaluated', 'مُقيّم') : t('Awaiting model', 'بانتظار النموذج')}</span></td></tr>`;
    }).join('')}</tbody></table></div><div class="table-footer"><span>${icon('clock')}${board.updated_at ? `${t('Updated', 'آخر تحديث')} ${date(board.updated_at)}` : t('Results appear after the first successful evaluation.', 'تظهر النتائج بعد أول تقييم ناجح.')}</span><button class="text-button" data-view="experiments">${t('All experiments', 'جميع التجارب')}${icon('arrow')}</button></div></section>
    <section class="card benchmark-card"><div class="card-heading"><h2>${t('Same rules. Fair results.', 'قواعد موحدة. نتائج عادلة.')}</h2>${icon('check')}</div><p>${t('Every submission runs through the same evaluation pipeline.', 'كل مشاركة تمر عبر مسار التقييم نفسه.')}</p><div class="split-bar" aria-label="70% train, 15% validation, 15% test"><span></span><span></span><span></span></div><div class="split-labels"><span><i></i>${t('Train', 'تدريب')}<strong>70%</strong></span><span><i></i>${t('Validation', 'تحقق')}<strong>15%</strong></span><span><i></i>${t('Test', 'اختبار')}<strong>15%</strong></span></div><ul class="check-list"><li>${icon('check')}${t('Fixed, duplicate-aware split', 'تقسيم ثابت يراعي التكرار')}</li><li>${icon('check')}${t('Automated CPU evaluation', 'تقييم تلقائي على CPU')}</li><li>${icon('check')}${t('Accuracy first. Macro-F1 breaks ties.', 'الدقة أولًا. Macro-F1 يحسم التعادل.')}</li></ul><button class="text-button" data-view="protocol">${t('Read the protocol', 'اقرأ المنهجية')}${icon('arrow')}</button></section></div>
    <section class="card recent-card"><div class="card-heading"><div><h2>${t('Latest experiments', 'أحدث التجارب')}</h2><p>${t('From a new idea to a verified result.', 'من فكرة جديدة إلى نتيجة موثقة.')}</p></div><button class="text-button" data-view="experiments">${t('View all', 'عرض الكل')}${icon('arrow')}</button></div>${board.results.filter(r => r.phase === phase).length ? resultTable(board.results.filter(r => r.phase === phase).slice().sort((a, b) => b.evaluated_at.localeCompare(a.evaluated_at)).slice(0, 4)) : empty(t('The next breakthrough starts with a first run.', 'الإنجاز القادم يبدأ بالتجربة الأولى.'), t('Submit a trained model to start the comparison. Only real, independently evaluated results appear here.', 'أضف نموذجًا مدرّبًا لبدء المقارنة. لا تظهر هنا إلا النتائج الحقيقية المحسوبة بالتقييم.'))}</section>`;
}
const stat = (i: string, value: string, label: string, sub: string) => `<div class="stat-card"><div class="stat-top"><span>${label}</span>${icon(i)}</div><strong>${value}</strong><small>${escape(sub)}</small></div>`;
function resultTable(results: Result[]) {
  if (!results.length) return empty(t('No experiments in this view', 'لا توجد تجارب في هذا العرض'), t('Choose another filter or submit a model to get started.', 'اختر تصفية أخرى أو أضف نموذجًا للبدء.'));
  return `<div class="table-wrap"><table><thead><tr><th>${t('Model', 'النموذج')}</th><th>${t('Researcher', 'الباحث')}</th><th>${t('Accuracy', 'الدقة')}</th><th>Macro-F1</th><th>${t('Evaluated', 'التقييم')}</th><th>${t('Status', 'الحالة')}</th><th><span class="sr-only">${t('Details', 'التفاصيل')}</span></th></tr></thead><tbody>${results.map(r => `<tr><td><button class="result-name" data-details="${escape(resultKey(r))}">${escape(r.model_name)}</button><small class="sub-id">${escape(r.id)}</small></td><td>${escape(name(r.participant))}</td><td class="metric">${percent(r.accuracy)}</td><td class="metric">${percent(r.macro_f1)}</td><td class="date-cell">${date(r.evaluated_at)}</td><td><span class="status ${r.status === 'success' ? 'success' : 'failed'}">${r.status === 'success' ? t('Evaluated', 'مُقيّم') : t('Failed', 'فشل')}</span></td><td><button class="icon-button" data-details="${escape(resultKey(r))}" aria-label="${t('Details for', 'تفاصيل')} ${escape(r.model_name)}">${icon('arrow')}</button></td></tr>`).join('')}</tbody></table></div>`;
}
function experiments() {
  const results = board.results.filter(r => r.phase === phase && (participantFilter === 'all' || r.participant === participantFilter) && `${r.model_name} ${r.id}`.toLowerCase().includes(search.toLowerCase())).sort((a, b) => b.evaluated_at.localeCompare(a.evaluated_at));
  return `<section class="card"><div class="filters"><label class="search-field">${icon('search')}<input id="experiment-search" type="search" placeholder="${t('Search models…', 'ابحث عن نموذج…')}" aria-label="${t('Search models', 'البحث عن النماذج')}" value="${escape(search)}" /></label><label class="sr-only" for="participant-filter">${t('Researcher', 'الباحث')}</label><select id="participant-filter"><option value="all">${t('All researchers', 'جميع الباحثين')}</option>${board.participants.map(p => `<option value="${escape(p.id)}" ${participantFilter === p.id ? 'selected' : ''}>${escape(name(p.id))}</option>`).join('')}</select>${phaseSwitch()}</div><div id="experiment-results">${resultTable(results)}</div><div class="table-footer">${t(`${results.length} experiment${results.length === 1 ? '' : 's'}`, `${results.length} تجربة`)}<span>${t('Metrics are computed by the evaluation pipeline.', 'يحسب مسار التقييم جميع المقاييس.')}</span></div></section>`;
}
function compare() {
  const candidates = sorted();
  selected = selected.filter(id => candidates.some(r => resultKey(r) === id));
  const chosen = selected.map(id => candidates.find(r => resultKey(r) === id)!);
  return `<section class="card"><div class="card-heading"><div><h2>${t('Side by side', 'مقارنة مباشرة')}</h2><p>${t('Select up to three evaluated models from the same phase.', 'اختر حتى ثلاثة نماذج مُقيّمة من المرحلة نفسها.')}</p></div>${phaseSwitch()}</div>${!candidates.length ? empty(t('A comparison needs evaluated models.', 'المقارنة تحتاج نماذج مُقيّمة.'), t('After models complete evaluation, compare accuracy, Macro-F1, and per-class recall here. Final comparisons require the same test split.', 'بعد اكتمال التقييم يمكنك مقارنة الدقة وMacro-F1 واستدعاء كل فئة هنا. تتطلب المقارنة النهائية تقسيم اختبار موحدًا.')) : `<div class="comparison-picker">${candidates.map(r => `<label class="pick-model ${selected.includes(resultKey(r)) ? 'selected' : ''}"><input type="checkbox" data-compare="${escape(resultKey(r))}" ${selected.includes(resultKey(r)) ? 'checked' : ''} ${selected.length >= 3 && !selected.includes(resultKey(r)) ? 'disabled' : ''}/><span><strong>${escape(r.model_name)}</strong><small>${escape(name(r.participant))} · ${percent(r.accuracy)}</small></span></label>`).join('')}</div>${chosen.length ? `<div class="comparison-grid">${chosen.map(r => `<div class="compare-card">${avatar(r.participant)}<h3>${escape(r.model_name)}</h3><p>${escape(name(r.participant))}</p><div class="compare-score">${percent(r.accuracy)}<small>${t('Accuracy', 'الدقة')}</small></div><dl><div><dt>Macro-F1</dt><dd>${percent(r.macro_f1)}</dd></div><div><dt>${t('Balanced accuracy', 'الدقة المتوازنة')}</dt><dd>${percent(r.balanced_accuracy)}</dd></div></dl><button class="text-button" data-details="${escape(resultKey(r))}">${t('Inspect result', 'تفاصيل النتيجة')}${icon('arrow')}</button></div>`).join('')}</div><div class="recall-comparison"><h3>${t('Per-class recall', 'استدعاء كل فئة')}</h3>${board.classes.map(label => `<div class="recall-row"><span>${escape(label)}</span><div>${chosen.map((r, i) => { const recall = r.per_class?.find(c => c.label === label)?.recall; return `<div class="bar-row"><span class="bar-track"><i class="bar-color-${i}" style="width:${Math.max(0, Math.min(100, (recall || 0) * 100))}%"></i></span><span>${percent(recall)}</span><small>${escape(r.model_name)}</small></div>`; }).join('')}</div></div>`).join('')}</div>` : empty(t('Choose models above', 'اختر النماذج أعلاه'), t('Select at least one model to inspect its metrics.', 'اختر نموذجًا واحدًا على الأقل لعرض مقاييسه.'), false)}`}</section>`;
}
function protocol() {
  return `<div class="protocol-intro"><span class="section-icon">${icon('book')}</span><div><h2>${t('A benchmark you can reproduce.', 'معيار يمكنك إعادة تجربته.')}</h2><p>${t('Shuaa studies 12-class solar module classification from infrared imagery. The models run on a server or laptop; the benchmark does not impose an edge-device size constraint.', 'يدرس شعاع تصنيف الألواح الشمسية إلى 12 فئة باستخدام الصور الحرارية. تعمل النماذج على سيرفر أو لابتوب، ولا يفرض المعيار قيود حجم خاصة بالأجهزة الصغيرة.')}</p></div></div><div class="protocol-grid"><section class="card padded"><span class="number-label">01 / ${t('DATA', 'البيانات')}</span><h2>InfraredSolarModules</h2><p>${t('20,000 infrared images released by Raptor Maps. The dataset contains normal modules and 11 anomaly categories; half of the images are normal.', '20,000 صورة حرارية نشرها Raptor Maps. تشمل ألواحًا سليمة و11 فئة شذوذ؛ نصف الصور لألواح سليمة.')}</p><p>${t('19,988 images are eligible after excluding 12 images with conflicting labels in exact duplicate pairs. Only exact duplicates were checked.', 'تبقّت 19,988 صورة بعد استبعاد 12 صورة ذات تسميات متعارضة ضمن أزواج متطابقة. فُحص التكرار المطابق فقط.')}</p>${link('https://github.com/RaptorMaps/InfraredSolarModules', t('Explore the original dataset', 'استعرض البيانات الأصلية'))}<div class="category-tags">${board.classes.map(c => `<span>${escape(c)}</span>`).join('')}</div></section><section class="card padded"><span class="number-label">02 / ${t('SPLIT', 'التقسيم')}</span><h2>${t('Separate learning from evaluation.', 'نفصل التعلم عن التقييم.')}</h2><div class="protocol-split"><div><strong>70%</strong><span>${t('Train', 'تدريب')}</span></div><div><strong>15%</strong><span>${t('Validation', 'تحقق')}</span></div><div><strong>15%</strong><span>${t('Test', 'اختبار')}</span></div></div><p>${t('Use one versioned split and a fixed seed. Identical images stay together. Augmentation applies only to training data. The published split manifest is the source of truth for exact counts.', 'نستخدم تقسيمًا موحدًا بإصدار وبذرة ثابتة. تبقى الصور المتطابقة معًا. تطبق زيادة البيانات على التدريب فقط. ملف التقسيم المنشور هو مرجع الأعداد الفعلية.')}</p></section><section class="card padded"><span class="number-label">03 / ${t('EVALUATION', 'التقييم')}</span><h2>${t('Measured, never self-reported.', 'نتائج محسوبة، وليست مدخلة يدويًا.')}</h2><p>${t('The pipeline runs ONNX inference on CPU using the declared preprocessing and fixed class order. Rankings use accuracy, followed by Macro-F1. Equal scores share a rank.', 'يشغّل المسار استدلال ONNX على CPU باستخدام المعالجة المعلنة وترتيب الفئات الثابت. الترتيب بالدقة ثم Macro-F1، والنتائج المتساوية تشترك في المركز.')}</p><p>${t('Validation guides experiments. Final test results are published only after one model per researcher is locked. Test data are publicly available, so this is a methodological holdout, not a secret test.', 'توجّه مجموعة التحقق التجارب. لا تنشر نتائج الاختبار النهائي إلا بعد تثبيت نموذج لكل باحث. بيانات الاختبار عامة، لذا حجزها منهجي وليس اختبارًا سريًا.')}</p></section><section class="card padded"><span class="number-label">04 / ${t('RESEARCH', 'البحث العلمي')}</span><h2>${t('Architecture with a paper trail.', 'معماريات موثّقة علميًا.')}</h2><div class="paper"><span class="paper-tag">ICML 2021</span><h3>EfficientNetV2</h3><p>${t('Smaller Models and Faster Training — Tan & Le. EfficientNetV2-S is a planned transfer-learning baseline.', 'نماذج أصغر وتدريب أسرع — Tan وLe. النموذج EfficientNetV2-S هو خط أساس مخطط لنقل التعلم.')}</p>${link('https://proceedings.mlr.press/v139/tan21a.html', t('Read the research paper', 'اقرأ الورقة العلمية'))}</div><div class="paper"><span class="paper-tag">CVPR 2023</span><h3>ConvNeXt V2</h3><p>${t('Co-designing and Scaling ConvNets with Masked Autoencoders — Woo et al. ConvNeXtV2-Tiny is the second planned candidate.', 'التصميم والتوسيع المشترك للشبكات الالتفافية مع المشفرات الذاتية المقنّعة — Woo وآخرون. النموذج ConvNeXtV2-Tiny هو المرشح الثاني المخطط.')}</p>${link('https://arxiv.org/abs/2301.00808', t('Read the research paper', 'اقرأ الورقة العلمية'))}</div></section></div><div class="notice">${t('No architecture is declared the winner in advance. The best model is the one supported by completed, reproducible experiments.', 'لا نعلن فوز أي معمارية مسبقًا. أفضل نموذج هو ما تدعمه التجارب المكتملة القابلة لإعادة التنفيذ.')}</div>`;
}
function submit() {
  return `<div class="submit-layout"><section class="card padded"><div class="submission-heading"><span class="section-icon">${icon('upload')}</span><div><h2>${t('From your notebook to the leaderboard.', 'من دفتر التدريب إلى لوحة الترتيب.')}</h2><p>${t('Submit through the shared GitHub repository.', 'أضف مشاركتك عبر مستودع GitHub المشترك.')}</p></div></div><ol class="steps"><li><span>01</span><div><h3>${t('Train on the shared split', 'درّب على التقسيم المشترك')}</h3><p>${t('Use the versioned training and validation split. Keep the test partition out of training and model selection. Record your architecture, seed, and paper.', 'استخدم تقسيم التدريب والتحقق الموحّد. أبقِ الاختبار خارج التدريب واختيار النموذج. سجّل المعمارية والبذرة والورقة العلمية.')}</p></div></li><li><span>02</span><div><h3>${t('Export and publish your weights', 'صدّر الأوزان وانشرها')}</h3><p>${t('Export an ONNX model with the fixed 12-class order. Publish it as a GitHub Release asset and calculate its SHA-256 checksum.', 'صدّر نموذج ONNX بترتيب الفئات الاثنتي عشرة الثابت. انشره كمرفق في GitHub Release واحسب بصمة SHA-256.')}</p></div></li><li><span>03</span><div><h3>${t('Add your submission manifest', 'أضف ملف تعريف المشاركة')}</h3><p>${t('Copy the repository template. Include your researcher ID, model name, code and paper links, preprocessing, split version, and weights URL and checksum.', 'انسخ قالب المستودع. أضف معرّف الباحث واسم النموذج وروابط الكود والورقة والمعالجة وإصدار التقسيم ورابط الأوزان وبصمتها.')}</p></div></li><li><span>04</span><div><h3>${t('Merge. Evaluate. Compare.', 'ادمج. قيّم. قارن.')}</h3><p>${t('After the submission is merged, GitHub Actions evaluates it and publishes the result. Check the workflow logs if evaluation fails; earlier successful results remain available.', 'بعد دمج المشاركة، يقيّمها GitHub Actions وينشر النتيجة. راجع سجلات التشغيل عند الفشل؛ تبقى النتائج الناجحة السابقة متاحة.')}</p></div></li></ol><div class="submission-actions"><a class="primary-button" href="${repo}/tree/main/submissions" target="_blank" rel="noopener noreferrer">${icon('github')}${t('Open submission templates', 'افتح قوالب المشاركة')}${icon('external')}</a><a class="secondary-button" href="${repo}/actions" target="_blank" rel="noopener noreferrer">${t('View workflow runs', 'عرض عمليات التشغيل')}${icon('external')}</a></div></section><aside><section class="card padded"><h2>${t('Submission essentials', 'متطلبات المشاركة')}</h2><ul class="check-list spacious"><li>${icon('check')}${t('One of the three team researcher IDs', 'معرّف أحد باحثي الفريق الثلاثة')}</li><li>${icon('check')}${t('A traceable research paper', 'ورقة علمية قابلة للتتبع')}</li><li>${icon('check')}${t('ONNX weights with a verified checksum', 'أوزان ONNX ببصمة موثّقة')}</li><li>${icon('check')}${t('Fixed split and class order', 'تقسيم وترتيب فئات ثابتان')}</li><li>${icon('check')}${t('Explicit preprocessing settings', 'إعدادات معالجة واضحة')}</li></ul></section><div class="submission-note">${icon('book')}<h3>${t('Reproducibility is part of the score.', 'قابلية إعادة التجربة جزء من البحث.')}</h3><p>${t('Keep your training configuration, notebook, and checkpoints alongside the experiment. Report what ran, not what was planned.', 'احتفظ بإعدادات التدريب والدفتر ونقاط الاستكمال مع التجربة. وثّق ما نُفّذ فعلًا.')}</p></div></aside></div>`;
}
function details(id: string) {
  const r = board.results.find(r => resultKey(r) === id);
  if (!r) return '';
  return `<div class="modal-backdrop" data-action="backdrop"><section class="modal" role="dialog" aria-modal="true" aria-labelledby="result-title"><div class="modal-heading"><div><div class="eyebrow">${escape(name(r.participant))} / ${r.phase === 'validation' ? t('VALIDATION', 'التحقق') : t('FINAL TEST', 'الاختبار النهائي')}</div><h2 id="result-title">${escape(r.model_name)}</h2><p>${date(r.evaluated_at)}</p></div><button class="icon-button modal-close" data-action="close" aria-label="${t('Close details', 'إغلاق التفاصيل')}">${icon('close')}</button></div>${r.status === 'failed' ? `<div class="notice error"><strong>${t('Evaluation failed', 'فشل التقييم')}</strong><p>${escape(r.error || t('See the GitHub Actions logs for details.', 'راجع سجلات GitHub Actions للتفاصيل.'))}</p></div>` : `<div class="modal-stats">${stat('trophy', percent(r.accuracy), t('Accuracy', 'الدقة'), '')}${stat('activity', percent(r.macro_f1), 'Macro-F1', '')}${stat('layers', percent(r.balanced_accuracy), t('Balanced accuracy', 'الدقة المتوازنة'), '')}</div>${hyperparameters(r.training)}${binaryDetails(r)}<h3>${t('Per-class performance', 'أداء كل فئة')}</h3>${r.per_class?.length ? `<div class="table-wrap"><table><thead><tr><th>${t('Class', 'الفئة')}</th><th>${t('Precision', 'الإحكام')}</th><th>${t('Recall', 'الاستدعاء')}</th><th>F1</th><th>${t('Support', 'عدد العينات')}</th></tr></thead><tbody>${r.per_class.map(c => `<tr><td>${escape(c.label)}</td><td>${percent(c.precision)}</td><td>${percent(c.recall)}</td><td>${percent(c.f1)}</td><td>${c.support}</td></tr>`).join('')}</tbody></table></div>` : `<p class="muted">${t('Per-class metrics are not available for this result.', 'مقاييس الفئات غير متاحة لهذه النتيجة.')}</p>`}<h3>${t('Confusion matrix', 'مصفوفة الالتباس')}</h3><p class="muted">${t('Rows: actual class · Columns: predicted class', 'الصفوف: الفئة الفعلية · الأعمدة: الفئة المتوقعة')}</p>${r.confusion_matrix?.length ? `<div class="table-wrap matrix-wrap"><table class="matrix"><thead><tr><th></th>${board.classes.map(c => `<th>${escape(c)}</th>`).join('')}</tr></thead><tbody>${r.confusion_matrix.map((row, i) => `<tr><th>${escape(board.classes[i] || String(i))}</th>${row.map((value, j) => `<td class="${i === j ? 'diagonal' : ''}" style="--intensity:${Math.min(0.85, 0.06 + value / Math.max(1, ...row) * 0.6)}">${value}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : `<p class="muted">${t('Confusion matrix not available.', 'مصفوفة الالتباس غير متاحة.')}</p>`}`}
    <h3>${t('Provenance', 'مصدر النتيجة')}</h3><dl class="provenance"><div><dt>${t('Result ID', 'معرّف النتيجة')}</dt><dd>${escape(r.id)}</dd></div><div><dt>${t('Source commit', 'نسخة الكود')}</dt><dd>${escape(r.source_commit || '—')}</dd></div><div><dt>Checkpoint SHA-256</dt><dd>${escape(r.checkpoint_sha256 || '—')}</dd></div><div><dt>${t('Split hash', 'بصمة التقسيم')}</dt><dd>${escape(r.split_hash || '—')}</dd></div></dl><div class="detail-links">${link(r.paper_url, t('Research paper', 'الورقة العلمية'))}${link(r.code_url, t('Training code', 'كود التدريب'))}</div></section></div>`;
}

function hyperparameters(training?: Training) {
  if (!training?.components.length) return `<h3>${t('Hyperparameters', 'المعلمات الفائقة')}</h3><p class="muted">${t('Training settings were not supplied with this result.', 'لم تُرفق إعدادات التدريب مع هذه النتيجة.')}</p>`;
  const method = training.ensemble_method === 'probability_mean'
    ? t('Equal probability mean', 'متوسط احتمالات متساوٍ')
    : t('Single model', 'نموذج واحد');
  return `<h3>${t('Hyperparameters', 'المعلمات الفائقة')}</h3><p class="muted">${t('Selection: ', 'الاختيار: ')}${method}</p><div class="table-wrap"><table><thead><tr><th>${t('Component', 'المكوّن')}</th><th>${t('Completed epochs', 'العصور المكتملة')}</th><th>${t('Optimizer', 'المُحسِّن')}</th><th>${t('Initial learning rate', 'معدل التعلم الابتدائي')}</th></tr></thead><tbody>${training.components.map(component => `<tr><td><strong>${escape(component.model_name)}</strong><small class="sub-id">${escape(component.run_id)} · ${t('seed', 'البذرة')} ${component.seed}</small></td><td>${component.epochs_completed} / ${component.configured_epochs}</td><td>${escape(component.optimizer)}</td><td>${learningRate(component.initial_learning_rate)}</td></tr>`).join('')}</tbody></table></div>`;
}

function binaryDetails(result: Result) {
  const matrix = result.confusion_matrix;
  const normal = board.classes.indexOf('No-Anomaly');
  if (!matrix || normal < 0 || matrix.length !== board.classes.length || matrix.some(row => row.length !== board.classes.length || row.some(value => !Number.isInteger(value) || value < 0))) return '';
  const tn = matrix[normal][normal];
  const fp = matrix[normal].reduce((sum, value) => sum + value, 0) - tn;
  const fn = matrix.reduce((sum, row) => sum + row[normal], 0) - tn;
  const total = matrix.flat().reduce((sum, value) => sum + value, 0);
  const tp = total - tn - fp - fn;
  if (!total || !Number.isFinite(tp) || tp < 0) return '';
  return `<h3>${t('Healthy vs defective', 'سليم أم معيب')}</h3>
    <p class="muted">${t('The top predicted class from the 12-class model is grouped as healthy (No-Anomaly) or defective (any anomaly). These are results on the selected evaluation phase, not a separately trained binary model.', 'نجمع الفئة الأعلى في النموذج ذي 12 فئة إلى سليم (No-Anomaly) أو معيب (أي فئة شذوذ). هذه نتيجة مرحلة التقييم المختارة، وليست نموذجًا ثنائيًا دُرّب منفصلًا.')}</p>
    <div class="modal-stats">${stat('check', percent((tn + tp) / total), t('Binary accuracy', 'دقة سليم/معيب'), `${tn + tp} / ${total}`)}${stat('activity', percent(tp / (tp + fn)), t('Defect recall', 'استدعاء العيوب'), `${tp} / ${tp + fn}`)}${stat('layers', percent(tn / (tn + fp)), t('Healthy recall', 'استدعاء السليم'), `${tn} / ${tn + fp}`)}</div>
    <div class="table-wrap"><table><thead><tr><th>${t('Actual', 'الفعلي')}</th><th>${t('Predicted healthy', 'توقع سليم')}</th><th>${t('Predicted defective', 'توقع معيب')}</th></tr></thead><tbody><tr><th>${t('Healthy', 'سليم')}</th><td>${tn}</td><td>${fp}</td></tr><tr><th>${t('Defective', 'معيب')}</th><td>${fn}</td><td>${tp}</td></tr></tbody></table></div>`;
}

async function load() {
  loading = true; loadError = false; shell();
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}data/leaderboard.json`, { cache: 'no-cache' });
    if (!response.ok) throw new Error('Unavailable');
    const data: unknown = await response.json();
    if (!data || typeof data !== 'object') throw new Error('Invalid data');
    const b = data as Board;
    if (b.schema_version !== 1 || !Array.isArray(b.results) || !Array.isArray(b.classes) || !Array.isArray(b.participants)) throw new Error('Invalid schema');
    board = b;
  } catch { loadError = true; }
  loading = false; shell();
}
document.addEventListener('click', event => {
  const element = (event.target as Element).closest<HTMLElement>('button, [data-action]');
  if (!element) return;
  if (element.dataset.view) { view = element.dataset.view as View; menuOpen = false; shell(); window.scrollTo(0, 0); document.querySelector<HTMLElement>('#main-content')?.focus({ preventScroll: true }); }
  else if (element.dataset.phase) { phase = element.dataset.phase as Phase; selected = []; shell(); }
  else if (element.dataset.details) { modalId = element.dataset.details; shell(); document.body.style.overflow = 'hidden'; }
  else if (element.dataset.action === 'language') { lang = lang === 'en' ? 'ar' : 'en'; try { localStorage.setItem('shuaa-language', lang); } catch { /* Private browsing may disable storage. */ } shell(); }
  else if (element.dataset.action === 'reload') void load();
  else if (element.dataset.action === 'menu') { menuOpen = !menuOpen; shell(); }
  else if (element.dataset.action === 'close' || (element.dataset.action === 'backdrop' && event.target === element)) closeModal();
});
function closeModal() { const id = modalId; modalId = null; document.body.style.overflow = ''; shell(); Array.from(document.querySelectorAll<HTMLButtonElement>('[data-details]')).find(b => b.dataset.details === id)?.focus(); }
document.addEventListener('change', event => {
  const target = event.target as HTMLInputElement;
  if (target.id === 'participant-filter') { participantFilter = target.value; shell(); }
  if (target.dataset.compare) { selected = target.checked ? [...selected, target.dataset.compare].slice(0, 3) : selected.filter(id => id !== target.dataset.compare); shell(); }
});
document.addEventListener('input', event => {
  const target = event.target as HTMLInputElement;
  if (target.id === 'experiment-search') { search = target.value; const position = target.selectionStart; shell(); const field = document.querySelector<HTMLInputElement>('#experiment-search'); field?.focus(); try { field?.setSelectionRange(position, position); } catch { /* Search input selection is browser-dependent. */ } }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menuOpen) { menuOpen = false; shell(); document.querySelector<HTMLButtonElement>('.mobile-menu')?.focus(); }
  if (!modalId) return;
  if (event.key === 'Escape') closeModal();
  if (event.key === 'Tab') {
    const focusable = Array.from(document.querySelectorAll<HTMLElement>('.modal button, .modal a[href], .modal [tabindex="0"]'));
    const first = focusable[0], last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  }
});
void load();
