import React from "react";
import ReactDOM from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { AppStateProvider } from "./contexts/AppStateContext";
import { FeaturesPage } from "@/routes/features/page";
import { RootPage } from "./routes/page";
import { AttentionHeadPage } from "./routes/attn-heads/page";
import { DictionaryPage } from "./routes/dictionaries/page";
import { ModelsPage } from "./routes/models/page";
import BookmarksPage from "./routes/bookmarks/page";
import { CircuitsPage } from "./routes/circuits/page";
import { SearchCircuitsPage } from "./routes/search-circuits/page";
import ThreeDVisualPage from "./routes/3D-visualization/page";
import { PlayGamePage } from "./routes/play-game/page";
import { PlotGraphPage } from "./routes/plot-graph/page";
import { LogitLensPage } from "./routes/logit-lens/page";
import { TacticFeaturesPage } from "./routes/tactic-features/page";
import { GlobalWeightPage } from "./routes/global-weight/page";
import { FunctionalMicrocircuitPage } from "./routes/functional-microcircuit/page";
import { PositionFeaturePage } from "./routes/position-feature/page";
import { FeatureInteractionPage } from "./routes/feature-interaction/page";
import { InteractionCircuitPage } from "./routes/interaction-circuit/page";
import { CircuitTaxonomyPage } from "./routes/circuit-taxonomy/page";
import { SemanticSupernodeGraphPage } from "./routes/semantic-supernode-graph/page";


const router = createBrowserRouter([
  {
    path: "/features",
    element: <FeaturesPage />,
  },
  {
    path: "/dictionaries",
    element: <DictionaryPage />,
  },
  {
    path: "/attn-heads",
    element: <AttentionHeadPage />,
  },
  {
    path: "/models",
    element: <ModelsPage />,
  },
  {
    path: "/bookmarks",
    element: <BookmarksPage />,
  },
  {
    path: "/circuits",
    element: <CircuitsPage />,
  },
  {
    path: "/3D-visualization",
    element: <ThreeDVisualPage />,
  },
  {
    path: "/play-game",
    element: <PlayGamePage />,
  },
  {
    path: "/search-circuits",
    element: <SearchCircuitsPage />,
  },
  {
    path: "/plot-graph",
    element: <PlotGraphPage />,
  },
  {
    path: "/logit-lens",
    element: <LogitLensPage />,
  },
  {
    path: "/tactic-features",
    element: <TacticFeaturesPage />,
  },
  {
    path: "/global-weight",
    element: <GlobalWeightPage />,
  },
  {
    path: "/functional-microcircuit",
    element: <FunctionalMicrocircuitPage />,
  },
  {
    path: "/position-feature",
    element: <PositionFeaturePage />,
  },
  {
    path: "/feature-interaction",
    element: <FeatureInteractionPage />,
  },
  {
    path: "/interaction-circuit",
    element: <InteractionCircuitPage />,
  },
  {
    path: "/circuit-taxonomy",
    element: <CircuitTaxonomyPage />,
  },
  {
    path: "/semantic-supernode-graph",
    element: <SemanticSupernodeGraphPage />,
  },
  {
    path: "/",
    element: <RootPage />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppStateProvider>
      <RouterProvider router={router} />
    </AppStateProvider>
  </React.StrictMode>
);
