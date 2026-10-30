import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('the dashboard renders every section from the generated results', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Every cycle');
  for (const name of [
    /camera box reports/,
    /camera moves/,
    /new station costs/,
    /people anonymised/,
    /no PyTorch/,
    /written blind/,
  ]) {
    await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('switching the recording redraws the timeline', async ({ page }) => {
  await page.goto('/');
  const tl = page.getByRole('img', { name: /cycles happened/ });
  const before = await tl.getAttribute('aria-label');
  await page.getByRole('button', { name: 'Camera moved' }).click();
  await expect(tl).not.toHaveAttribute('aria-label', before!);
});

test('no serious accessibility violations', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  const a = await new AxeBuilder({ page } as unknown as ConstructorParameters<typeof AxeBuilder>[0])
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();
  expect(a.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')).toEqual([]);
});
