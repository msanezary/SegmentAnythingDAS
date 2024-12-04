# SegmentAnythingDAS
```
SegmentAnythingDAS
├─ code_1.py
├─ cudatest.py
├─ fenetre.py
├─ interface.py
├─ labels.txt
├─ README.md
├─ sam2
│  ├─ .clang-format
│  ├─ .watchmanconfig
│  ├─ assets
│  │  ├─ model_diagram.png
│  │  └─ sa_v_dataset.jpg
│  ├─ backend.Dockerfile
│  ├─ checkpoints
│  │  └─ download_ckpts.sh
│  ├─ CODE_OF_CONDUCT.md
│  ├─ CONTRIBUTING.md
│  ├─ demo
│  │  ├─ backend
│  │  │  └─ server
│  │  │     ├─ app.py
│  │  │     ├─ app_conf.py
│  │  │     ├─ data
│  │  │     │  ├─ data_types.py
│  │  │     │  ├─ loader.py
│  │  │     │  ├─ resolver.py
│  │  │     │  ├─ schema.py
│  │  │     │  ├─ store.py
│  │  │     │  └─ transcoder.py
│  │  │     └─ inference
│  │  │        ├─ data_types.py
│  │  │        ├─ multipart.py
│  │  │        └─ predictor.py
│  │  ├─ data
│  │  │  └─ gallery
│  │  │     ├─ 01_dog.mp4
│  │  │     ├─ 02_cups.mp4
│  │  │     ├─ 03_blocks.mp4
│  │  │     ├─ 04_coffee.mp4
│  │  │     └─ 05_default_juggle.mp4
│  │  ├─ frontend
│  │  │  ├─ .babelrc
│  │  │  ├─ .dockerignore
│  │  │  ├─ .eslintignore
│  │  │  ├─ .eslintrc.cjs
│  │  │  ├─ .prettierignore
│  │  │  ├─ .prettierrc.json
│  │  │  ├─ .watchmanconfig
│  │  │  ├─ frontend.Dockerfile
│  │  │  ├─ index.html
│  │  │  ├─ package.json
│  │  │  ├─ postcss.config.js
│  │  │  ├─ public
│  │  │  │  └─ fonts
│  │  │  │     └─ Inter-VariableFont_opsz,wght.ttf
│  │  │  ├─ schema.graphql
│  │  │  ├─ schemas
│  │  │  │  ├─ inference-api-schema.graphql
│  │  │  │  ├─ merge-schemas.ts
│  │  │  │  └─ video-api-schema.graphql
│  │  │  ├─ src
│  │  │  │  ├─ App.tsx
│  │  │  │  ├─ assets
│  │  │  │  │  ├─ icons
│  │  │  │  │  │  ├─ angery.png
│  │  │  │  │  │  ├─ heart.png
│  │  │  │  │  │  └─ whistle.png
│  │  │  │  │  ├─ scss
│  │  │  │  │  │  └─ App.scss
│  │  │  │  │  └─ videos
│  │  │  │  │     ├─ sam2_720px_dark.mp4
│  │  │  │  │     └─ sam2_video_poster.png
│  │  │  │  ├─ common
│  │  │  │  │  ├─ codecs
│  │  │  │  │  │  ├─ VideoDecoder.ts
│  │  │  │  │  │  ├─ VideoEncoder.ts
│  │  │  │  │  │  └─ WebCodecUtils.ts
│  │  │  │  │  ├─ components
│  │  │  │  │  │  ├─ annotations
│  │  │  │  │  │  │  ├─ AddObjectButton.tsx
│  │  │  │  │  │  │  ├─ ClearAllPointsInVideoButton.tsx
│  │  │  │  │  │  │  ├─ CloseSessionButton.tsx
│  │  │  │  │  │  │  ├─ FirstClickView.tsx
│  │  │  │  │  │  │  ├─ LimitNotice.tsx
│  │  │  │  │  │  │  ├─ MobileObjectsList.tsx
│  │  │  │  │  │  │  ├─ MobileObjectsToolbar.tsx
│  │  │  │  │  │  │  ├─ MobileObjectsToolbarHeader.tsx
│  │  │  │  │  │  │  ├─ ObjectActions.tsx
│  │  │  │  │  │  │  ├─ ObjectPlaceholder.tsx
│  │  │  │  │  │  │  ├─ ObjectsToolbar.tsx
│  │  │  │  │  │  │  ├─ ObjectsToolbarBottomActions.tsx
│  │  │  │  │  │  │  ├─ ObjectsToolbarHeader.tsx
│  │  │  │  │  │  │  ├─ ObjectThumbnail.tsx
│  │  │  │  │  │  │  ├─ ObjectUtils.ts
│  │  │  │  │  │  │  ├─ PointsToggle.tsx
│  │  │  │  │  │  │  ├─ PrimaryCTAButton.tsx
│  │  │  │  │  │  │  ├─ ToolbarObject.tsx
│  │  │  │  │  │  │  ├─ ToolbarObjectContainer.tsx
│  │  │  │  │  │  │  ├─ TrackletsAnnotation.tsx
│  │  │  │  │  │  │  ├─ TrackletSwimlane.tsx
│  │  │  │  │  │  │  └─ useTracklets.ts
│  │  │  │  │  │  ├─ button
│  │  │  │  │  │  │  ├─ GradientBorder.tsx
│  │  │  │  │  │  │  ├─ PlaybackButton.tsx
│  │  │  │  │  │  │  ├─ PrimaryCTAButton.tsx
│  │  │  │  │  │  │  ├─ ResponsiveButton.tsx
│  │  │  │  │  │  │  └─ TrackAndPlayButton.tsx
│  │  │  │  │  │  ├─ code
│  │  │  │  │  │  │  └─ InitializeLocalMonaco.ts
│  │  │  │  │  │  ├─ effects
│  │  │  │  │  │  │  ├─ BackgroundEffects.tsx
│  │  │  │  │  │  │  ├─ EffectsCarousel.tsx
│  │  │  │  │  │  │  ├─ EffectsCarouselShadow.tsx
│  │  │  │  │  │  │  ├─ EffectsToolbar.tsx
│  │  │  │  │  │  │  ├─ EffectsToolbarBottomActions.tsx
│  │  │  │  │  │  │  ├─ EffectsToolbarHeader.tsx
│  │  │  │  │  │  │  ├─ EffectsUtils.ts
│  │  │  │  │  │  │  ├─ EffectVariantBadge.tsx
│  │  │  │  │  │  │  ├─ HighlightEffects.tsx
│  │  │  │  │  │  │  ├─ MobileEffectsToolbar.tsx
│  │  │  │  │  │  │  └─ MoreFunEffects.tsx
│  │  │  │  │  │  ├─ gallery
│  │  │  │  │  │  │  ├─ ChangeVideoModal.tsx
│  │  │  │  │  │  │  ├─ DefaultVideoGalleryModalTrigger.tsx
│  │  │  │  │  │  │  ├─ DemoVideoGallery.tsx
│  │  │  │  │  │  │  ├─ DemoVideoGalleryModal.tsx
│  │  │  │  │  │  │  ├─ useUploadVideo.ts
│  │  │  │  │  │  │  ├─ VideoGalleryUploadPhoto.tsx
│  │  │  │  │  │  │  ├─ VideoPhoto.tsx
│  │  │  │  │  │  │  └─ __generated__
│  │  │  │  │  │  │     ├─ DemoVideoGalleryModalQuery.graphql.ts
│  │  │  │  │  │  │     ├─ DemoVideoGalleryQuery.graphql.ts
│  │  │  │  │  │  │     └─ useUploadVideoMutation.graphql.ts
│  │  │  │  │  │  ├─ icons
│  │  │  │  │  │  │  └─ GitHubIcon.tsx
│  │  │  │  │  │  ├─ MobileFirstClickBanner.tsx
│  │  │  │  │  │  ├─ options
│  │  │  │  │  │  │  ├─ DownloadOption.tsx
│  │  │  │  │  │  │  ├─ GalleryOption.tsx
│  │  │  │  │  │  │  ├─ MoreOptionsToolbar.tsx
│  │  │  │  │  │  │  ├─ MoreOptionsToolbarBottomActions.tsx
│  │  │  │  │  │  │  ├─ OptionButton.tsx
│  │  │  │  │  │  │  ├─ ShareSection.tsx
│  │  │  │  │  │  │  ├─ ShareUtils.ts
│  │  │  │  │  │  │  ├─ TryAnotherVideoSection.tsx
│  │  │  │  │  │  │  ├─ UploadOption.tsx
│  │  │  │  │  │  │  ├─ useDownloadVideo.ts
│  │  │  │  │  │  │  └─ __generated__
│  │  │  │  │  │  │     └─ GetLinkOptionShareVideoMutation.graphql.ts
│  │  │  │  │  │  ├─ session
│  │  │  │  │  │  │  ├─ RestartSessionButton.tsx
│  │  │  │  │  │  │  ├─ useCloseSessionBeforeUnload.ts
│  │  │  │  │  │  │  ├─ useRestartSession.ts
│  │  │  │  │  │  │  └─ __generated__
│  │  │  │  │  │  │     └─ useCloseSessionBeforeUnloadMutation.graphql.ts
│  │  │  │  │  │  ├─ snackbar
│  │  │  │  │  │  │  ├─ DemoMessagesSnackbarUtils.ts
│  │  │  │  │  │  │  ├─ MessagesSnackbar.tsx
│  │  │  │  │  │  │  ├─ snackbarAtoms.ts
│  │  │  │  │  │  │  ├─ useDemoMessagesSnackbar.ts
│  │  │  │  │  │  │  ├─ useExpireMessage.ts
│  │  │  │  │  │  │  └─ useMessagesSnackbar.ts
│  │  │  │  │  │  ├─ toolbar
│  │  │  │  │  │  │  ├─ DesktopToolbar.tsx
│  │  │  │  │  │  │  ├─ MobileToolbar.tsx
│  │  │  │  │  │  │  ├─ Toolbar.tsx
│  │  │  │  │  │  │  ├─ ToolbarActionIcon.tsx
│  │  │  │  │  │  │  ├─ ToolbarBottomActionsWrapper.tsx
│  │  │  │  │  │  │  ├─ ToolbarConfig.tsx
│  │  │  │  │  │  │  ├─ ToolbarHeaderWrapper.tsx
│  │  │  │  │  │  │  ├─ ToolbarProgressChip.tsx
│  │  │  │  │  │  │  ├─ ToolbarSection.tsx
│  │  │  │  │  │  │  ├─ useListenToStreamingState.ts
│  │  │  │  │  │  │  └─ useToolbarTabs.ts
│  │  │  │  │  │  ├─ Tooltip.tsx
│  │  │  │  │  │  ├─ useFunctionThrottle.tsx
│  │  │  │  │  │  └─ video
│  │  │  │  │  │     ├─ ChangeVideoModal.tsx
│  │  │  │  │  │     ├─ editor
│  │  │  │  │  │     │  ├─ atoms.ts
│  │  │  │  │  │     │  ├─ DemoVideoEditor.tsx
│  │  │  │  │  │     │  ├─ ImageUtils.ts
│  │  │  │  │  │     │  ├─ useResetEditor.ts
│  │  │  │  │  │     │  ├─ useVideo.ts
│  │  │  │  │  │     │  ├─ useVideoEffect.ts
│  │  │  │  │  │     │  ├─ VideoEditor.tsx
│  │  │  │  │  │     │  └─ VideoEditorUtils.ts
│  │  │  │  │  │     ├─ effects
│  │  │  │  │  │     │  ├─ ArrowGLEffect.ts
│  │  │  │  │  │     │  ├─ BackgroundBlurEffect.ts
│  │  │  │  │  │     │  ├─ BackgroundTextEffect.ts
│  │  │  │  │  │     │  ├─ BaseGLEffect.ts
│  │  │  │  │  │     │  ├─ BurstGLEffect.ts
│  │  │  │  │  │     │  ├─ CutoutGLEffect.ts
│  │  │  │  │  │     │  ├─ DesaturateEffect.ts
│  │  │  │  │  │     │  ├─ Effect.ts
│  │  │  │  │  │     │  ├─ Effects.ts
│  │  │  │  │  │     │  ├─ EffectUtils.ts
│  │  │  │  │  │     │  ├─ EraseBackgroundEffect.ts
│  │  │  │  │  │     │  ├─ EraseForegroundEffect.ts
│  │  │  │  │  │     │  ├─ EraseForegroundGLEffect.ts
│  │  │  │  │  │     │  ├─ GradientEffect.ts
│  │  │  │  │  │     │  ├─ NoisyMaskEffect.ts
│  │  │  │  │  │     │  ├─ OriginalEffect.ts
│  │  │  │  │  │     │  ├─ OverlayEffect.ts
│  │  │  │  │  │     │  ├─ PixelateEffect.ts
│  │  │  │  │  │     │  ├─ PixelateMaskGLEffect.ts
│  │  │  │  │  │     │  ├─ ReplaceGLEffect.ts
│  │  │  │  │  │     │  ├─ ScopeGLEffect.ts
│  │  │  │  │  │     │  ├─ shaders
│  │  │  │  │  │     │  │  ├─ Arrow.frag
│  │  │  │  │  │     │  │  ├─ BackgroundBlur.frag
│  │  │  │  │  │     │  │  ├─ Burst.frag
│  │  │  │  │  │     │  │  ├─ Cutout.frag
│  │  │  │  │  │     │  │  ├─ DefaultVert.vert
│  │  │  │  │  │     │  │  ├─ EraseForeground.frag
│  │  │  │  │  │     │  │  ├─ Gradient.frag
│  │  │  │  │  │     │  │  ├─ NoisyMask.frag
│  │  │  │  │  │     │  │  ├─ Overlay.frag
│  │  │  │  │  │     │  │  ├─ Overlay.vert
│  │  │  │  │  │     │  │  ├─ Pixelate.frag
│  │  │  │  │  │     │  │  ├─ PixelateMask.frag
│  │  │  │  │  │     │  │  ├─ Replace.frag
│  │  │  │  │  │     │  │  ├─ Scope.frag
│  │  │  │  │  │     │  │  ├─ Sobel.frag
│  │  │  │  │  │     │  │  └─ VibrantMask.frag
│  │  │  │  │  │     │  ├─ SobelEffect.ts
│  │  │  │  │  │     │  └─ VibrantMaskEffect.ts
│  │  │  │  │  │     ├─ EventEmitter.ts
│  │  │  │  │  │     ├─ filmstrip
│  │  │  │  │  │     │  ├─ atoms.ts
│  │  │  │  │  │     │  ├─ FilmstripUtil.tsx
│  │  │  │  │  │     │  ├─ SelectedFrameHelper.ts
│  │  │  │  │  │     │  ├─ useDisableScrolling.ts
│  │  │  │  │  │     │  ├─ useSelectedFrameHelper.ts
│  │  │  │  │  │     │  └─ VideoFilmstrip.tsx
│  │  │  │  │  │     ├─ layers
│  │  │  │  │  │     │  ├─ InteractionLayer.tsx
│  │  │  │  │  │     │  └─ PointsLayer.tsx
│  │  │  │  │  │     ├─ useInputVideo.ts
│  │  │  │  │  │     ├─ useVideoWorker.ts
│  │  │  │  │  │     ├─ Video.tsx
│  │  │  │  │  │     ├─ VideoFilmstripWithPlayback.tsx
│  │  │  │  │  │     ├─ VideoLoadingOverlay.tsx
│  │  │  │  │  │     ├─ VideoWorker.ts
│  │  │  │  │  │     ├─ VideoWorkerBridge.ts
│  │  │  │  │  │     ├─ VideoWorkerContext.ts
│  │  │  │  │  │     └─ VideoWorkerTypes.ts
│  │  │  │  │  ├─ error
│  │  │  │  │  │  ├─ ErrorFallback.tsx
│  │  │  │  │  │  ├─ ErrorReport.tsx
│  │  │  │  │  │  ├─ errorReportAtom.ts
│  │  │  │  │  │  ├─ ErrorSerializationUtils.ts
│  │  │  │  │  │  ├─ ErrorUtils.ts
│  │  │  │  │  │  └─ useReportError.tsx
│  │  │  │  │  ├─ loading
│  │  │  │  │  │  ├─ LoadingMessage.tsx
│  │  │  │  │  │  ├─ LoadingStateScreen.tsx
│  │  │  │  │  │  ├─ StaticVideoPlayer.tsx
│  │  │  │  │  │  └─ UploadLoadingScreen.tsx
│  │  │  │  │  ├─ logger
│  │  │  │  │  │  ├─ DemoLogger.ts
│  │  │  │  │  │  ├─ LogEnvironment.ts
│  │  │  │  │  │  └─ Logger.ts
│  │  │  │  │  ├─ screen
│  │  │  │  │  │  └─ useScreenSize.tsx
│  │  │  │  │  ├─ tracker
│  │  │  │  │  │  ├─ SAM2Model.ts
│  │  │  │  │  │  ├─ Tracker.ts
│  │  │  │  │  │  ├─ Trackers.ts
│  │  │  │  │  │  ├─ TrackerTypes.ts
│  │  │  │  │  │  └─ __generated__
│  │  │  │  │  │     ├─ SAM2ModelAddNewPointsMutation.graphql.ts
│  │  │  │  │  │     ├─ SAM2ModelCancelPropagateInVideoMutation.graphql.ts
│  │  │  │  │  │     ├─ SAM2ModelClearPointsInFrameMutation.graphql.ts
│  │  │  │  │  │     ├─ SAM2ModelClearPointsInVideoMutation.graphql.ts
│  │  │  │  │  │     ├─ SAM2ModelCloseSessionMutation.graphql.ts
│  │  │  │  │  │     ├─ SAM2ModelRemoveObjectMutation.graphql.ts
│  │  │  │  │  │     └─ SAM2ModelStartSessionMutation.graphql.ts
│  │  │  │  │  └─ utils
│  │  │  │  │     ├─ emptyFunction.ts
│  │  │  │  │     ├─ FileUtils.ts
│  │  │  │  │     ├─ ImageUtils.ts
│  │  │  │  │     ├─ MaskUtils.ts
│  │  │  │  │     ├─ MultipartStream.ts
│  │  │  │  │     ├─ ShaderUtils.ts
│  │  │  │  │     └─ uuid.ts
│  │  │  │  ├─ debug
│  │  │  │  │  └─ stats
│  │  │  │  │     ├─ Stats.ts
│  │  │  │  │     └─ StatsView.tsx
│  │  │  │  ├─ demo
│  │  │  │  │  ├─ atoms.ts
│  │  │  │  │  ├─ DemoConfig.tsx
│  │  │  │  │  ├─ DemoErrorFallback.tsx
│  │  │  │  │  ├─ DemoSuspenseFallback.tsx
│  │  │  │  │  └─ SAM2DemoApp.tsx
│  │  │  │  ├─ graphql
│  │  │  │  │  ├─ errors
│  │  │  │  │  │  ├─ CreateFilmstripError.ts
│  │  │  │  │  │  ├─ DrawFrameError.ts
│  │  │  │  │  │  └─ WebGLContextError.ts
│  │  │  │  │  ├─ fetchGraphQL.ts
│  │  │  │  │  ├─ RelayEnvironment.ts
│  │  │  │  │  └─ RelayEnvironmentProvider.tsx
│  │  │  │  ├─ jscocotools
│  │  │  │  │  └─ mask.ts
│  │  │  │  ├─ layouts
│  │  │  │  │  ├─ DemoPageLayout.tsx
│  │  │  │  │  └─ RootLayout.tsx
│  │  │  │  ├─ main.tsx
│  │  │  │  ├─ routes
│  │  │  │  │  ├─ DemoPage.tsx
│  │  │  │  │  ├─ DemoPageWrapper.tsx
│  │  │  │  │  ├─ PageNotFoundPage.tsx
│  │  │  │  │  └─ __generated__
│  │  │  │  │     └─ DemoPageQuery.graphql.ts
│  │  │  │  ├─ settings
│  │  │  │  │  ├─ ApprovableInput.tsx
│  │  │  │  │  ├─ SAM2Settings.tsx
│  │  │  │  │  ├─ SettingsContextProvider.tsx
│  │  │  │  │  ├─ SettingsModal.tsx
│  │  │  │  │  ├─ SettingsReducer.ts
│  │  │  │  │  └─ useSettingsContext.tsx
│  │  │  │  ├─ theme
│  │  │  │  │  ├─ colors.ts
│  │  │  │  │  ├─ gradientStyle.ts
│  │  │  │  │  └─ tokens.stylex.ts
│  │  │  │  ├─ types
│  │  │  │  │  └─ mp4box
│  │  │  │  │     └─ index.d.ts
│  │  │  │  └─ vite-env.d.ts
│  │  │  ├─ tailwind.config.js
│  │  │  ├─ tsconfig.json
│  │  │  ├─ tsconfig.node.json
│  │  │  ├─ vite.config.ts
│  │  │  └─ yarn.lock
│  │  └─ README.md
│  ├─ docker-compose.yaml
│  ├─ INSTALL.md
│  ├─ LICENSE
│  ├─ LICENSE_cctorch
│  ├─ MANIFEST.in
│  ├─ notebooks
│  │  ├─ automatic_mask_generator_example.ipynb
│  │  ├─ images
│  │  │  ├─ cars.jpg
│  │  │  ├─ groceries.jpg
│  │  │  └─ truck.jpg
│  │  ├─ image_predictor_example.ipynb
│  │  ├─ videos
│  │  │  ├─ bedroom
│  │  │  │  ├─ 00000.jpg
│  │  │  │  ├─ 00001.jpg
│  │  │  │  ├─ 00002.jpg
│  │  │  │  ├─ 00003.jpg
│  │  │  │  ├─ 00004.jpg
│  │  │  │  ├─ 00005.jpg
│  │  │  │  ├─ 00006.jpg
│  │  │  │  ├─ 00007.jpg
│  │  │  │  ├─ 00008.jpg
│  │  │  │  ├─ 00009.jpg
│  │  │  │  ├─ 00010.jpg
│  │  │  │  ├─ 00011.jpg
│  │  │  │  ├─ 00012.jpg
│  │  │  │  ├─ 00013.jpg
│  │  │  │  ├─ 00014.jpg
│  │  │  │  ├─ 00015.jpg
│  │  │  │  ├─ 00016.jpg
│  │  │  │  ├─ 00017.jpg
│  │  │  │  ├─ 00018.jpg
│  │  │  │  ├─ 00019.jpg
│  │  │  │  ├─ 00020.jpg
│  │  │  │  ├─ 00021.jpg
│  │  │  │  ├─ 00022.jpg
│  │  │  │  ├─ 00023.jpg
│  │  │  │  ├─ 00024.jpg
│  │  │  │  ├─ 00025.jpg
│  │  │  │  ├─ 00026.jpg
│  │  │  │  ├─ 00027.jpg
│  │  │  │  ├─ 00028.jpg
│  │  │  │  ├─ 00029.jpg
│  │  │  │  ├─ 00030.jpg
│  │  │  │  ├─ 00031.jpg
│  │  │  │  ├─ 00032.jpg
│  │  │  │  ├─ 00033.jpg
│  │  │  │  ├─ 00034.jpg
│  │  │  │  ├─ 00035.jpg
│  │  │  │  ├─ 00036.jpg
│  │  │  │  ├─ 00037.jpg
│  │  │  │  ├─ 00038.jpg
│  │  │  │  ├─ 00039.jpg
│  │  │  │  ├─ 00040.jpg
│  │  │  │  ├─ 00041.jpg
│  │  │  │  ├─ 00042.jpg
│  │  │  │  ├─ 00043.jpg
│  │  │  │  ├─ 00044.jpg
│  │  │  │  ├─ 00045.jpg
│  │  │  │  ├─ 00046.jpg
│  │  │  │  ├─ 00047.jpg
│  │  │  │  ├─ 00048.jpg
│  │  │  │  ├─ 00049.jpg
│  │  │  │  ├─ 00050.jpg
│  │  │  │  ├─ 00051.jpg
│  │  │  │  ├─ 00052.jpg
│  │  │  │  ├─ 00053.jpg
│  │  │  │  ├─ 00054.jpg
│  │  │  │  ├─ 00055.jpg
│  │  │  │  ├─ 00056.jpg
│  │  │  │  ├─ 00057.jpg
│  │  │  │  ├─ 00058.jpg
│  │  │  │  ├─ 00059.jpg
│  │  │  │  ├─ 00060.jpg
│  │  │  │  ├─ 00061.jpg
│  │  │  │  ├─ 00062.jpg
│  │  │  │  ├─ 00063.jpg
│  │  │  │  ├─ 00064.jpg
│  │  │  │  ├─ 00065.jpg
│  │  │  │  ├─ 00066.jpg
│  │  │  │  ├─ 00067.jpg
│  │  │  │  ├─ 00068.jpg
│  │  │  │  ├─ 00069.jpg
│  │  │  │  ├─ 00070.jpg
│  │  │  │  ├─ 00071.jpg
│  │  │  │  ├─ 00072.jpg
│  │  │  │  ├─ 00073.jpg
│  │  │  │  ├─ 00074.jpg
│  │  │  │  ├─ 00075.jpg
│  │  │  │  ├─ 00076.jpg
│  │  │  │  ├─ 00077.jpg
│  │  │  │  ├─ 00078.jpg
│  │  │  │  ├─ 00079.jpg
│  │  │  │  ├─ 00080.jpg
│  │  │  │  ├─ 00081.jpg
│  │  │  │  ├─ 00082.jpg
│  │  │  │  ├─ 00083.jpg
│  │  │  │  ├─ 00084.jpg
│  │  │  │  ├─ 00085.jpg
│  │  │  │  ├─ 00086.jpg
│  │  │  │  ├─ 00087.jpg
│  │  │  │  ├─ 00088.jpg
│  │  │  │  ├─ 00089.jpg
│  │  │  │  ├─ 00090.jpg
│  │  │  │  ├─ 00091.jpg
│  │  │  │  ├─ 00092.jpg
│  │  │  │  ├─ 00093.jpg
│  │  │  │  ├─ 00094.jpg
│  │  │  │  ├─ 00095.jpg
│  │  │  │  ├─ 00096.jpg
│  │  │  │  ├─ 00097.jpg
│  │  │  │  ├─ 00098.jpg
│  │  │  │  ├─ 00099.jpg
│  │  │  │  ├─ 00100.jpg
│  │  │  │  ├─ 00101.jpg
│  │  │  │  ├─ 00102.jpg
│  │  │  │  ├─ 00103.jpg
│  │  │  │  ├─ 00104.jpg
│  │  │  │  ├─ 00105.jpg
│  │  │  │  ├─ 00106.jpg
│  │  │  │  ├─ 00107.jpg
│  │  │  │  ├─ 00108.jpg
│  │  │  │  ├─ 00109.jpg
│  │  │  │  ├─ 00110.jpg
│  │  │  │  ├─ 00111.jpg
│  │  │  │  ├─ 00112.jpg
│  │  │  │  ├─ 00113.jpg
│  │  │  │  ├─ 00114.jpg
│  │  │  │  ├─ 00115.jpg
│  │  │  │  ├─ 00116.jpg
│  │  │  │  ├─ 00117.jpg
│  │  │  │  ├─ 00118.jpg
│  │  │  │  ├─ 00119.jpg
│  │  │  │  ├─ 00120.jpg
│  │  │  │  ├─ 00121.jpg
│  │  │  │  ├─ 00122.jpg
│  │  │  │  ├─ 00123.jpg
│  │  │  │  ├─ 00124.jpg
│  │  │  │  ├─ 00125.jpg
│  │  │  │  ├─ 00126.jpg
│  │  │  │  ├─ 00127.jpg
│  │  │  │  ├─ 00128.jpg
│  │  │  │  ├─ 00129.jpg
│  │  │  │  ├─ 00130.jpg
│  │  │  │  ├─ 00131.jpg
│  │  │  │  ├─ 00132.jpg
│  │  │  │  ├─ 00133.jpg
│  │  │  │  ├─ 00134.jpg
│  │  │  │  ├─ 00135.jpg
│  │  │  │  ├─ 00136.jpg
│  │  │  │  ├─ 00137.jpg
│  │  │  │  ├─ 00138.jpg
│  │  │  │  ├─ 00139.jpg
│  │  │  │  ├─ 00140.jpg
│  │  │  │  ├─ 00141.jpg
│  │  │  │  ├─ 00142.jpg
│  │  │  │  ├─ 00143.jpg
│  │  │  │  ├─ 00144.jpg
│  │  │  │  ├─ 00145.jpg
│  │  │  │  ├─ 00146.jpg
│  │  │  │  ├─ 00147.jpg
│  │  │  │  ├─ 00148.jpg
│  │  │  │  ├─ 00149.jpg
│  │  │  │  ├─ 00150.jpg
│  │  │  │  ├─ 00151.jpg
│  │  │  │  ├─ 00152.jpg
│  │  │  │  ├─ 00153.jpg
│  │  │  │  ├─ 00154.jpg
│  │  │  │  ├─ 00155.jpg
│  │  │  │  ├─ 00156.jpg
│  │  │  │  ├─ 00157.jpg
│  │  │  │  ├─ 00158.jpg
│  │  │  │  ├─ 00159.jpg
│  │  │  │  ├─ 00160.jpg
│  │  │  │  ├─ 00161.jpg
│  │  │  │  ├─ 00162.jpg
│  │  │  │  ├─ 00163.jpg
│  │  │  │  ├─ 00164.jpg
│  │  │  │  ├─ 00165.jpg
│  │  │  │  ├─ 00166.jpg
│  │  │  │  ├─ 00167.jpg
│  │  │  │  ├─ 00168.jpg
│  │  │  │  ├─ 00169.jpg
│  │  │  │  ├─ 00170.jpg
│  │  │  │  ├─ 00171.jpg
│  │  │  │  ├─ 00172.jpg
│  │  │  │  ├─ 00173.jpg
│  │  │  │  ├─ 00174.jpg
│  │  │  │  ├─ 00175.jpg
│  │  │  │  ├─ 00176.jpg
│  │  │  │  ├─ 00177.jpg
│  │  │  │  ├─ 00178.jpg
│  │  │  │  ├─ 00179.jpg
│  │  │  │  ├─ 00180.jpg
│  │  │  │  ├─ 00181.jpg
│  │  │  │  ├─ 00182.jpg
│  │  │  │  ├─ 00183.jpg
│  │  │  │  ├─ 00184.jpg
│  │  │  │  ├─ 00185.jpg
│  │  │  │  ├─ 00186.jpg
│  │  │  │  ├─ 00187.jpg
│  │  │  │  ├─ 00188.jpg
│  │  │  │  ├─ 00189.jpg
│  │  │  │  ├─ 00190.jpg
│  │  │  │  ├─ 00191.jpg
│  │  │  │  ├─ 00192.jpg
│  │  │  │  ├─ 00193.jpg
│  │  │  │  ├─ 00194.jpg
│  │  │  │  ├─ 00195.jpg
│  │  │  │  ├─ 00196.jpg
│  │  │  │  ├─ 00197.jpg
│  │  │  │  ├─ 00198.jpg
│  │  │  │  └─ 00199.jpg
│  │  │  └─ bedroom.mp4
│  │  └─ video_predictor_example.ipynb
│  ├─ pyproject.toml
│  ├─ README.md
│  ├─ sam2
│  │  ├─ automatic_mask_generator.py
│  │  ├─ build_sam.py
│  │  ├─ configs
│  │  │  ├─ sam2
│  │  │  │  ├─ sam2_hiera_b+.yaml
│  │  │  │  ├─ sam2_hiera_l.yaml
│  │  │  │  ├─ sam2_hiera_s.yaml
│  │  │  │  └─ sam2_hiera_t.yaml
│  │  │  ├─ sam2.1
│  │  │  │  ├─ sam2.1_hiera_b+.yaml
│  │  │  │  ├─ sam2.1_hiera_l.yaml
│  │  │  │  ├─ sam2.1_hiera_s.yaml
│  │  │  │  └─ sam2.1_hiera_t.yaml
│  │  │  └─ sam2.1_training
│  │  │     └─ sam2.1_hiera_b+_MOSE_finetune.yaml
│  │  ├─ csrc
│  │  │  └─ connected_components.cu
│  │  ├─ modeling
│  │  │  ├─ backbones
│  │  │  │  ├─ hieradet.py
│  │  │  │  ├─ image_encoder.py
│  │  │  │  ├─ utils.py
│  │  │  │  └─ __init__.py
│  │  │  ├─ memory_attention.py
│  │  │  ├─ memory_encoder.py
│  │  │  ├─ position_encoding.py
│  │  │  ├─ sam
│  │  │  │  ├─ mask_decoder.py
│  │  │  │  ├─ prompt_encoder.py
│  │  │  │  ├─ transformer.py
│  │  │  │  └─ __init__.py
│  │  │  ├─ sam2_base.py
│  │  │  ├─ sam2_utils.py
│  │  │  └─ __init__.py
│  │  ├─ sam2_hiera_b+.yaml
│  │  ├─ sam2_hiera_l.yaml
│  │  ├─ sam2_hiera_s.yaml
│  │  ├─ sam2_hiera_t.yaml
│  │  ├─ sam2_image_predictor.py
│  │  ├─ sam2_video_predictor.py
│  │  ├─ utils
│  │  │  ├─ amg.py
│  │  │  ├─ misc.py
│  │  │  ├─ transforms.py
│  │  │  └─ __init__.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ build_sam.cpython-312.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ SAM_2.egg-info
│  │  ├─ dependency_links.txt
│  │  ├─ PKG-INFO
│  │  ├─ requires.txt
│  │  ├─ SOURCES.txt
│  │  └─ top_level.txt
│  ├─ sav_dataset
│  │  ├─ example
│  │  │  ├─ sav_000001.mp4
│  │  │  ├─ sav_000001_auto.json
│  │  │  └─ sav_000001_manual.json
│  │  ├─ LICENSE
│  │  ├─ LICENSE_DAVIS
│  │  ├─ LICENSE_VOS_BENCHMARK
│  │  ├─ README.md
│  │  ├─ requirements.txt
│  │  ├─ sav_evaluator.py
│  │  ├─ sav_visualization_example.ipynb
│  │  └─ utils
│  │     ├─ sav_benchmark.py
│  │     └─ sav_utils.py
│  ├─ setup.py
│  ├─ tools
│  │  ├─ README.md
│  │  └─ vos_inference.py
│  └─ training
│     ├─ assets
│     │  ├─ MOSE_sample_train_list.txt
│     │  └─ MOSE_sample_val_list.txt
│     ├─ dataset
│     │  ├─ sam2_datasets.py
│     │  ├─ transforms.py
│     │  ├─ utils.py
│     │  ├─ vos_dataset.py
│     │  ├─ vos_raw_dataset.py
│     │  ├─ vos_sampler.py
│     │  ├─ vos_segment_loader.py
│     │  └─ __init__.py
│     ├─ loss_fns.py
│     ├─ model
│     │  ├─ sam2.py
│     │  └─ __init__.py
│     ├─ optimizer.py
│     ├─ README.md
│     ├─ scripts
│     │  └─ sav_frame_extraction_submitit.py
│     ├─ train.py
│     ├─ trainer.py
│     ├─ utils
│     │  ├─ checkpoint_utils.py
│     │  ├─ data_utils.py
│     │  ├─ distributed.py
│     │  ├─ logger.py
│     │  ├─ train_utils.py
│     │  └─ __init__.py
│     └─ __init__.py
└─ ToDo.txt

```