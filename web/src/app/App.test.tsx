import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import fixture from './fixture.json';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function stubFetch(impl: () => Promise<Partial<Response>>) {
  vi.stubGlobal('fetch', vi.fn(impl));
}

describe('App', () => {
  it('shows a skeleton while loading', () => {
    stubFetch(() => new Promise(() => {}));
    render(<App />);
    expect(screen.getByRole('status', { name: 'Loading results' })).toBeTruthy();
  });

  it('shows the empty state when there are no results', async () => {
    stubFetch(async () => ({ ok: false, status: 404 }));
    render(<App />);
    await waitFor(() => expect(screen.getByText('No results yet')).toBeTruthy());
  });

  it('shows an error with retry when the file is malformed', async () => {
    stubFetch(async () => ({ ok: true, status: 200, json: async () => ({ nope: true }) }));
    render(<App />);
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('not in the expected format'));
  });

  it('renders the timeline and every section from results', async () => {
    stubFetch(async () => ({ ok: true, status: 200, json: async () => fixture }));
    render(<App />);
    await waitFor(() => expect(screen.getByRole('heading', { level: 1 })).toBeTruthy());
    for (const id of ['process', 'shift', 'onboarding', 'perception', 'edge', 'heldout']) {
      expect(document.getElementById(`${id}-h`)).toBeTruthy();
    }
    expect(screen.getByRole('img', { name: /cycles happened/ })).toBeTruthy();
  });
});
