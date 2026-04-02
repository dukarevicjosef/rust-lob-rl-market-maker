export interface SimDisplayInfo {
  mode:           "simulate" | "replay";
  replayDate?:    string;
  replayEvents?:  number;
}

let _state: SimDisplayInfo = { mode: "simulate" };
const _listeners = new Set<() => void>();

export const simInfoStore = {
  get: (): SimDisplayInfo => _state,

  set: (info: SimDisplayInfo): void => {
    _state = info;
    _listeners.forEach((fn) => fn());
  },

  subscribe: (fn: () => void): (() => void) => {
    _listeners.add(fn);
    return () => _listeners.delete(fn);
  },
};
