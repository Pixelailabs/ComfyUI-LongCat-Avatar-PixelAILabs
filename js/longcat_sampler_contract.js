import { app } from "../../scripts/app.js";

const NODE_CLASS = "PixelAILabs_LongCatAvatar_Sampler";
const IMAGE_INPUT = "image";
const STAGE_WIDGET = "stage_1";
const STAGE_AI2V = "ai2v";
const STAGE_AT2V = "at2v";

const EFFECTIVE_WIDGETS = Object.freeze({
  steps: Object.freeze({ value: 8, index: 4 }),
  text_guidance_scale: Object.freeze({ value: 1.0, index: 5 }),
  audio_guidance_scale: Object.freeze({ value: 1.0, index: 6 }),
});

const enhancedNodes = new WeakSet();

function isNamedValues(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function isKnownSamplerArray(values) {
  return (
    Array.isArray(values) &&
    values.length >= 7 &&
    [STAGE_AI2V, STAGE_AT2V].includes(values[0]) &&
    ["480p", "720p"].includes(values[1])
  );
}

function migrateSamplerNodeData(nodeData) {
  if (!nodeData || nodeData.type !== NODE_CLASS) {
    return;
  }

  if (isKnownSamplerArray(nodeData.widgets_values)) {
    for (const { value, index } of Object.values(EFFECTIVE_WIDGETS)) {
      nodeData.widgets_values[index] = value;
    }
  }

  if (isNamedValues(nodeData.widgets_values_named)) {
    for (const [name, { value }] of Object.entries(EFFECTIVE_WIDGETS)) {
      nodeData.widgets_values_named[name] = value;
    }
  }
}

function migrateSamplerWorkflow(graphData) {
  const pending = [graphData];
  const visited = new Set();
  while (pending.length > 0) {
    const graph = pending.shift();
    if (!graph || visited.has(graph)) {
      continue;
    }
    visited.add(graph);
    for (const nodeData of graph?.nodes ?? []) {
      migrateSamplerNodeData(nodeData);
    }
    pending.push(...(graph?.definitions?.subgraphs ?? []));
  }
}

function synchronizeEffectiveWidget(node, widget, effectiveValue) {
  if (!widget) {
    return;
  }
  widget.options ??= {};
  widget.options.min = effectiveValue;
  widget.options.max = effectiveValue;
  widget.options.disabled = true;
  widget.value = effectiveValue;
  widget.serializeValue = () => effectiveValue;

  if (!widget.__longcatAvatarEffectiveCallback) {
    const originalCallback = widget.callback;
    widget.callback = function (_requestedValue, ...args) {
      widget.value = effectiveValue;
      const result = originalCallback?.call(this, effectiveValue, ...args);
      widget.value = effectiveValue;
      node.setDirtyCanvas?.(true, true);
      return result;
    };
    widget.__longcatAvatarEffectiveCallback = true;
  }
}

function synchronizeImageState(node) {
  const stage = node.widgets?.find((widget) => widget.name === STAGE_WIDGET)?.value;
  const imageInput = node.inputs?.find((input) => input.name === IMAGE_INPUT);
  if (!imageInput) {
    return;
  }
  imageInput.label =
    stage === STAGE_AT2V
      ? "image (optional for at2v)"
      : "image (required for ai2v)";
  node.setDirtyCanvas?.(true, true);
}

function synchronizeNode(node) {
  for (const [name, { value }] of Object.entries(EFFECTIVE_WIDGETS)) {
    const widget = node.widgets?.find((candidate) => candidate.name === name);
    synchronizeEffectiveWidget(node, widget, value);
  }
  synchronizeImageState(node);
}

function enhanceNode(node) {
  if (!node || node.comfyClass !== NODE_CLASS) {
    return;
  }
  if (enhancedNodes.has(node)) {
    synchronizeNode(node);
    return;
  }

  const stageWidget = node.widgets?.find((widget) => widget.name === STAGE_WIDGET);
  if (stageWidget && !stageWidget.__longcatAvatarStageCallback) {
    const originalCallback = stageWidget.callback;
    stageWidget.callback = function (...args) {
      const result = originalCallback?.apply(this, args);
      synchronizeImageState(node);
      return result;
    };
    stageWidget.__longcatAvatarStageCallback = true;
  }

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    try {
      return originalOnConfigure?.apply(this, args);
    } finally {
      synchronizeNode(this);
    }
  };

  const originalOnSerialize = node.onSerialize;
  node.onSerialize = function (nodeData, ...args) {
    synchronizeNode(this);
    const result = originalOnSerialize?.call(this, nodeData, ...args);
    migrateSamplerNodeData(nodeData);
    return result;
  };

  enhancedNodes.add(node);
  synchronizeNode(node);
}

app.registerExtension({
  name: "longcat-avatar.sampler-effective-contract",
  beforeConfigureGraph(graphData) {
    migrateSamplerWorkflow(graphData);
  },
  nodeCreated(node) {
    enhanceNode(node);
  },
  loadedGraphNode(node) {
    enhanceNode(node);
  },
});
