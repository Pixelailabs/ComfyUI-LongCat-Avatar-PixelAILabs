import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const AUDIO_CROP_NODE_CLASS = "PixelAILabs_LongCatAvatar_AudioCrop";
const AUDIO_WIDGET_NAME = "longcat_audio_crop_preview";
const AUDIO_WIDGET_TYPE = "audioUI";
const AUDIO_WIDGET_FACTORY = "AUDIO_UI";
const EXECUTE_WIDGET_NAME = "longcat_audio_crop_execute";
const PREVIEW_AUDIO_NODE_CLASS = "PreviewAudio";
const EMPTY_AUDIO_CLASS = "empty-audio-widget";

function audioResultURL(audio) {
  const params = new URLSearchParams();
  params.set("filename", audio?.filename ?? "");
  params.set("subfolder", audio?.subfolder ?? "");
  params.set("type", audio?.type ?? "temp");
  return api.apiURL(`/view?${params.toString()}`);
}

function updatePreview(widget, audioElement, output) {
  const audio = output?.audio?.[0];
  if (!audio?.filename) {
    audioElement.removeAttribute("src");
    audioElement.load?.();
    widget.value = "";
    widget.callback?.("");
    audioElement.classList.add(EMPTY_AUDIO_CLASS);
    return;
  }

  const url = audioResultURL(audio);
  audioElement.src = url;
  audioElement.load?.();
  // IMPORTANT: mirror the host audioUI value/callback contract so Vue Nodes and the right panel update too.
  widget.value = url;
  widget.callback?.(url);
  audioElement.classList.remove(EMPTY_AUDIO_CLASS);
}

function findDirectPreviewTargets(node) {
  const graph = node?.graph;
  const linkIds = node?.outputs?.[0]?.links ?? [];
  const targets = [];
  for (const linkId of linkIds) {
    const link = graph?.links?.[linkId];
    const target = link ? graph.getNodeById?.(link.target_id) : null;
    if (
      target?.comfyClass === PREVIEW_AUDIO_NODE_CLASS &&
      target.id !== null &&
      target.id !== undefined
    ) {
      targets.push(String(target.id));
    }
  }
  return [...new Set(targets)];
}

function addAudioCropPreview(node) {
  if (
    !node ||
    node.comfyClass !== AUDIO_CROP_NODE_CLASS ||
    node.__longcatAudioCropPreviewHooked
  ) {
    return;
  }

  const widget = node.widgets?.find(
    (candidate) =>
      candidate.name === AUDIO_WIDGET_NAME && candidate.type === AUDIO_WIDGET_TYPE,
  );
  const audio = widget?.element;
  if (!(audio instanceof HTMLAudioElement)) {
    console.warn("LongCat Audio Crop could not initialize the host AUDIO_UI preview widget.");
    return;
  }

  widget.serialize = false;
  widget.options ??= {};
  widget.options.serialize = false;
  widget.options.canvasOnly = false;
  audio.classList.add(EMPTY_AUDIO_CLASS);
  node.__longcatAudioCropPreviewHooked = true;
  node.__longcatClearAudioCropPreview = () => updatePreview(widget, audio, null);

  const previousOnExecuted = node.onExecuted;
  node.onExecuted = function (output) {
    previousOnExecuted?.call(this, output);
    updatePreview(widget, audio, output);
  };
}

function addCropPreviewButton(node) {
  if (
    !node ||
    node.comfyClass !== AUDIO_CROP_NODE_CLASS ||
    typeof node.addWidget !== "function" ||
    node.widgets?.some((widget) => widget.name === EXECUTE_WIDGET_NAME)
  ) {
    return;
  }

  const widget = node.addWidget("button", "Crop Preview", "", async () => {
    if (node.id === null || node.id === undefined) {
      console.warn("LongCat Audio Crop preview cannot run before the node has an id.");
      return;
    }

    node.__longcatClearAudioCropPreview?.();
    const previewTargets = findDirectPreviewTargets(node);
    if (previewTargets.length === 0) {
      console.warn(
        "LongCat Audio Crop preview requires a directly connected Core PreviewAudio node. No prompt was queued.",
      );
      return;
    }

    // IMPORTANT: target only directly connected Core PreviewAudio outputs; never fall through to model/video outputs.
    await app.queuePrompt(0, 1, previewTargets);
  });
  widget.name = EXECUTE_WIDGET_NAME;
  widget.serialize = false;
  widget.options ??= {};
  widget.options.serialize = false;
  widget.options.canvasOnly = true;
}

app.registerExtension({
  name: "longcat-avatar.audio-crop-preview",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const comfyClass = nodeType?.prototype?.comfyClass ?? nodeData?.name;
    if (comfyClass !== AUDIO_CROP_NODE_CLASS) {
      return;
    }

    // IMPORTANT: use Core's AUDIO_UI factory so its WidgetValueStore binding keeps Vue Nodes reactive.
    nodeData.input ??= {};
    nodeData.input.required ??= {};
    nodeData.input.required[AUDIO_WIDGET_NAME] = [AUDIO_WIDGET_FACTORY, {}];
  },
  nodeCreated(node) {
    addAudioCropPreview(node);
    addCropPreviewButton(node);
  },
  loadedGraphNode(node) {
    setTimeout(() => {
      addAudioCropPreview(node);
      addCropPreviewButton(node);
    }, 50);
  },
});
