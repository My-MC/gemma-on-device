import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";

export function AppSelect({
  value,
  onChange,
  disabled = false,
  labelId,
  options,
  className = "",
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  labelId: string;
  options: ReadonlyArray<{ value: string; label: string }>;
  className?: string;
}) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listboxRef = useRef<HTMLDivElement>(null);
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const [open, setOpen] = useState(false);
  const selectedOption = options[selectedIndex];

  useEffect(() => {
    if (!open) return;

    setActiveIndex(selectedIndex);
    listboxRef.current?.focus();
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnOutsideFocus = (event: FocusEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    document.addEventListener("focusin", closeOnOutsideFocus);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
      document.removeEventListener("focusin", closeOnOutsideFocus);
    };
  }, [open, selectedIndex]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (open) {
      document
        .getElementById(`${id}-option-${activeIndex}`)
        ?.scrollIntoView({ block: "nearest" });
    }
  }, [open, activeIndex, id]);

  const openMenu = () => {
    if (disabled) return;
    setActiveIndex(selectedIndex);
    setOpen(true);
  };

  const closeMenu = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  };

  const chooseOption = (index: number) => {
    const option = options[index];
    if (!option) return;
    onChange(option.value);
    closeMenu(true);
  };

  const handleListboxKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActiveIndex((index) =>
          Math.min(index + 1, Math.max(0, options.length - 1)),
        );
        break;
      case "ArrowUp":
        event.preventDefault();
        setActiveIndex((index) => Math.max(index - 1, 0));
        break;
      case "Home":
        event.preventDefault();
        setActiveIndex(0);
        break;
      case "End":
        event.preventDefault();
        setActiveIndex(Math.max(0, options.length - 1));
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        chooseOption(activeIndex);
        break;
      case "Escape":
        event.preventDefault();
        closeMenu(true);
        break;
      case "Tab":
        closeMenu(true);
        break;
    }
  };

  return (
    <div className={`relative w-full min-w-0 ${className}`} ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="group flex min-h-14 w-full min-w-0 items-center justify-between gap-4 rounded-xl border border-outline bg-surface-container-low px-4 text-left font-sans text-[0.9375rem] font-normal text-foreground focus-visible:outline-3 focus-visible:outline-primary/40 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-listbox`}
        aria-labelledby={`${labelId} ${id}-selected`}
        disabled={disabled || options.length === 0}
        onClick={() => (open ? closeMenu(false) : openMenu())}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            openMenu();
          }
        }}
      >
        <span
          id={`${id}-selected`}
          className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap"
        >
          {selectedOption?.label ?? "候補がありません"}
        </span>
        <span
          className="h-[9px] w-[9px] shrink-0 translate-y-[-2px] rotate-45 border-r-[1.5px] border-b-[1.5px] border-current transition-transform group-aria-expanded:translate-y-[2px] group-aria-expanded:rotate-[225deg]"
          aria-hidden="true"
        />
      </button>
      {open && (
        <div
          ref={listboxRef}
          id={`${id}-listbox`}
          className="absolute top-[calc(100%+8px)] left-0 z-20 max-h-[280px] w-full overflow-auto rounded-2xl border border-outline-variant bg-popover p-2 font-sans text-popover-foreground shadow-[0_8px_24px_rgb(16_24_40_/_16%)] outline-none focus-visible:outline-3 focus-visible:outline-primary/40 focus-visible:outline-offset-2 max-[480px]:w-full max-[480px]:max-w-[calc(100vw-40px)]"
          role="listbox"
          tabIndex={0}
          aria-labelledby={labelId}
          aria-activedescendant={`${id}-option-${activeIndex}`}
          onKeyDown={handleListboxKeyDown}
        >
          {options.map((option, index) => (
            <div
              id={`${id}-option-${index}`}
              key={option.value}
              className={`min-h-12 cursor-pointer rounded-[10px] px-4 py-3 text-sm leading-6 whitespace-normal [overflow-wrap:anywhere] ${index === selectedIndex ? "font-semibold text-primary" : "font-normal text-popover-foreground"} ${index === activeIndex ? "bg-secondary text-secondary-foreground" : "hover:bg-secondary hover:text-secondary-foreground"}`}
              role="option"
              tabIndex={-1}
              aria-selected={index === selectedIndex}
              data-active={index === activeIndex}
              onPointerDown={(event) => event.preventDefault()}
              onPointerMove={() => setActiveIndex(index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  event.stopPropagation();
                  chooseOption(index);
                }
              }}
              onClick={(event) => {
                event.stopPropagation();
                chooseOption(index);
              }}
            >
              {option.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
