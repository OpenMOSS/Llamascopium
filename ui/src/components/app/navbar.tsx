import { Link, useRouterState } from '@tanstack/react-router'
import { Settings } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * Application navigation bar with links to main sections.
 */
export const AppNavbar = () => {
  const routerState = useRouterState()
  const pathname = routerState.location.pathname

  return (
    <nav className="p-4 border-b border-border bg-card">
      <div className="container mx-auto flex flex-wrap items-center gap-x-8 gap-y-3">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-xl font-bold">SAE Visualizer</span>
        </Link>

        <div className="flex flex-wrap gap-4 items-center flex-1">
          <Link
            className={cn(
              'transition-colors hover:text-foreground/80 text-foreground/60',
              pathname.startsWith('/global-weights') &&
                'text-foreground font-medium',
            )}
            to="/global-weights"
          >
            Global weights
          </Link>
          <Link
            className={cn(
              'transition-colors hover:text-foreground/80 text-foreground/60',
              pathname.startsWith('/dictionaries') &&
                'text-foreground font-medium',
            )}
            to="/dictionaries"
          >
            Dictionaries
          </Link>
          <Link
            className={cn(
              'transition-colors hover:text-foreground/80 text-foreground/60',
              pathname.startsWith('/circuits') && 'text-foreground font-medium',
            )}
            to="/circuits"
          >
            Circuits
          </Link>
          <Link
            className={cn(
              'transition-colors hover:text-foreground/80 text-foreground/60',
              pathname.startsWith('/bookmarks') &&
                'text-foreground font-medium',
            )}
            to="/bookmarks"
          >
            Bookmarks
          </Link>
        </div>

        <Link
          className={cn(
            'flex items-center gap-1.5 transition-colors hover:text-foreground/80 text-foreground/60',
            pathname.startsWith('/admin') && 'text-foreground font-medium',
          )}
          to="/admin"
        >
          <Settings className="h-4 w-4" />
          Admin
        </Link>
      </div>
    </nav>
  )
}
