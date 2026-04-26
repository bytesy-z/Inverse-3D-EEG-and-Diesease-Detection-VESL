declare module 'plotly.js-dist-min' {
  export interface Data {
    x?: number[]
    y?: number[]
    mode?: 'lines' | 'markers' | 'lines+markers'
    name?: string
    line?: {
      color?: string
      width?: number
    }
    hoverinfo?: string
    yaxis?: string
    type?: string
    marker?: Record<string, unknown>
  }

  export interface Layout {
    paper_bgcolor?: string
    plot_bgcolor?: string
    font?: {
      color?: string
      size?: number
    }
    margin?: {
      l?: number
      r?: number
      t?: number
      b?: number
    }
    xaxis?: {
      title?: string
      titlefont?: { color?: string; size?: number }
      tickfont?: { color?: string; size?: number }
      gridcolor?: string
      zerolinecolor?: string
      zerolinewidth?: number
      dtick?: number
      range?: number[]
    }
    yaxis?: {
      title?: string
      tickfont?: { color?: string; size?: number }
      gridcolor?: string
      zerolinecolor?: string
      zerolinewidth?: number
      showticklabels?: boolean
      tickvals?: number[]
      ticktext?: string[]
      dtick?: number
      range?: number[]
    }
    showlegend?: boolean
    hovermode?: string
    dragmode?: string
    height?: number
  }

  export interface Config {
    responsive?: boolean
    displayModeBar?: boolean
    scrollZoom?: boolean
  }

  export function newPlot(
    el: HTMLElement | string,
    data: Data[],
    layout?: Partial<Layout>,
    config?: Partial<Config>
  ): Promise<void>

  export const Plots: {
    resize: (el: HTMLElement) => void
  }

  const Plotly: {
    newPlot: typeof newPlot
    Plots: typeof Plots
    Data: typeof Data
    Layout: typeof Layout
    Config: typeof Config
  }
  export default Plotly
}
