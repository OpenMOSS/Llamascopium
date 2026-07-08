import { cn } from "@/lib/utils";
import { Link, useLocation } from "react-router-dom";

export const AppNavbar = () => {
  const location = useLocation();

  return (
    <nav className="p-4">
      <div className="container mx-auto flex items-center gap-8">
        <img src="/openmoss.ico" alt="logo" className="h-8" />

        <div className="flex gap-4 items-center">
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/features" && "text-foreground"
            )}
            to="/features"
          >
            Features
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/dictionaries" && "text-foreground"
            )}
            to="/dictionaries"
          >
            Dictionaries
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/bookmarks" && "text-foreground"
            )}
            to="/bookmarks"
          >
            Bookmarks
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/models" && "text-foreground"
            )}
            to="/models"
          >
            Models
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/circuits" && "text-foreground"
            )}
            to="/circuits"
          >
            Circuits
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/search-circuits" && "text-foreground"
            )}
            to="/search-circuits"
          >
            Search Circuits
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/3D-visualization" && "text-foreground"
            )}
            to="/3D-visualization"
          >
            3D Visualization
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/play-game" && "text-foreground"
            )}
            to="/play-game"
          >
            Play Game
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/logit-lens" && "text-foreground"
            )}
            to="/logit-lens"
          >
            Logit Lens
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/tactic-features" && "text-foreground"
            )}
            to="/tactic-features"
          >
            Tactic Features
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/global-weight" && "text-foreground"
            )}
            to="/global-weight"
          >
            Global Weight
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/functional-microcircuit" && "text-foreground"
            )}
            to="/functional-microcircuit"
          >
            Functional Microcircuit
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/position-feature" && "text-foreground"
            )}
            to="/position-feature"
          >
            Position Feature
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/feature-interaction" && "text-foreground"
            )}
            to="/feature-interaction"
          >
            Feature Interaction
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/interaction-circuit" && "text-foreground"
            )}
            to="/interaction-circuit"
          >
            Interaction Circuit
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/circuit-taxonomy" && "text-foreground"
            )}
            to="/circuit-taxonomy"
          >
            Circuit Taxonomy
          </Link>
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/semantic-supernode-graph" && "text-foreground"
            )}
            to="/semantic-supernode-graph"
          >
            Semantic Supernode Graph
          </Link>
        </div>
      </div>
    </nav>
  );
};
