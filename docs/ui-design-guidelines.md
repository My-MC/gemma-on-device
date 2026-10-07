# UI Design Guidelines

Use this document when creating or changing screens in Gemma On Device. The design takes inspiration from Material 3's color roles, hierarchy, spacing, and shapes. It does not aim to reproduce official Material 3 components exactly; it prioritizes a consistent app experience that works well on desktop, mobile, and WSL environments.

## Core principles

- Use semantic CSS variables from `src/index.css` and Tailwind theme utilities instead of choosing colors per screen. Give backgrounds, cards, containers, text, borders, primary actions, and status colors distinct roles, and preserve those roles in both light and dark themes.
- Use the bundled Noto Sans JP font for Japanese text. Do not rely on operating-system or browser font settings for the visual result.
- Base spacing on multiples of 4px. Common steps are 8 / 12 / 16 / 24 / 32px; use them to express grouping and screen hierarchy.
- Choose shapes by role: about 12px for inputs, 16–28px for cards and list items, and pill shapes for chips and primary buttons.
- Make interactive areas easy to find and clearly indicate focus, disabled, and selected states without relying on color alone.

## Layout and responsive behavior

- On desktop, keep content to about 1120px wide and place primary navigation in a vertical rail on the left when the viewport is wider than 1000px.
- At 1000px and below, move navigation to a horizontal row at the top and let content use the available width.
- At 760px and below, fix navigation to the bottom and account for the safe area. On short viewports at 720px high or below, return fixed navigation to the normal flow so it does not crowd the content.
- At 480px and below, stack headings, action buttons, and multi-column inputs vertically. Verify that a 320px viewport does not cause horizontal scrolling.
- Reduce the number of columns in information cards, forms, and benchmark results on narrow screens. Let elements wrap or stack naturally instead of shrinking them until they are hard to read.

## Inputs and selection controls

- Do not use OS- or browser-rendered `<select>` popups. Some environments do not apply the bundled font to these controls, which can render Japanese text as tofu.
- Use the shared `AppSelect` in `src/AppSelect.tsx` for selection UI. Render the listbox in the DOM and expose `role="listbox"`, `role="option"`, selection state, and the active item. Support Arrow Up/Down, Home/End, Enter/Space, Escape, Tab, and outside clicks.
- Use 56px as the standard height for inputs and dropdown triggers. Match labels, borders, backgrounds, and focus rings to the theme tokens. Keep action buttons at least 48px high.
- Keep new selection controls browser-rendered, and verify keyboard interaction and focus restoration.

## Status and chips

- Show success, warning, and error states with foreground and container color pairs. Include text or another status cue rather than relying on a red/green color change alone.
- Use consistent dimensions for runtime, execution environment, and model status chips in the header. The current standard is 40px high on desktop and 36px at widths of 760px or less. Keep each chip's text on one line and allow wrapping between chips.

## Dialogs and long lists

- Give dialogs an opaque card surface so the screen behind them does not show through. Keep the scrim and dialog surface on separate layers.
- Keep the license dialog's visible area stable regardless of item count. The desktop target is up to 900px wide and 780px high (capped at 90% of the viewport); at widths of 600px or less, target 96% of the viewport height.
- Keep search and category controls in fixed positions while only the list scrolls. Verify that the dialog's outer dimensions stay stable for zero results, loading, error, one item, and many items.
- Preserve Escape-to-close behavior, keep focus inside the dialog, and return focus to the dialog trigger when it closes.

## Implementation rules

- Style components with Tailwind utilities in JSX. Do not add page-specific CSS classes or reintroduce `App.css` or `LicenseDialog.css`.
- Add shared foundations such as colors and fonts to the theme tokens in `src/index.css`. Use role-based utility names such as `bg-card` and `text-on-surface-variant` in component classes.
- Express exceptional dimensions and breakpoints with Tailwind arbitrary values and variants. Use the existing `short-mobile:` variant for navigation on short viewports.
- List complete class names in conditional branches for state-dependent styling. Do not assemble class names from fragments at runtime, where Tailwind may not detect them.
- Before adding a dependency or component library, check whether the existing shadcn/Radix-based components and Tailwind setup can meet the need.

## Checks when making changes

For frontend changes, run `bun run check` and `bun run build`. For layout changes, check at least 320 / 390 / 600 / 900 / 1280px widths for horizontal overflow, clipped text, visible focus, and navigation placement. At widths of 760px or less, also check a viewport height of 720px or less.

For selection UI changes, test opening, closing, selecting, and canceling with both mouse and keyboard. For dialog changes, filter content at 390px and 1280px widths and confirm that the outer dimensions remain stable when there are no results.
