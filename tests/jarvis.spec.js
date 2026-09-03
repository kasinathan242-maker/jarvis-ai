import { test, expect } from '@playwright/test';

test.describe('JARVIS AI E2E Tests', () => {

  test.beforeEach(async ({ page }) => {
    // Go to the JARVIS dashboard
    await page.goto('/');
  });

  test('should load the dashboard and check title/header', async ({ page }) => {
    await expect(page).toHaveTitle(/JARVIS AI/);
    
    // Check main header
    await expect(page.locator('.logo-title')).toHaveText('JARVIS AI');
    await expect(page.locator('.status-badge')).toContainText('SYSTEM ACTIVE');
    
    // Check main input area
    await expect(page.locator('#cmdInput')).toBeVisible();
    await expect(page.locator('.send-btn')).toBeVisible();
  });

  test('should check initial memory load', async ({ page }) => {
    const memoryList = page.locator('#memoryList');
    await expect(memoryList).toBeVisible();
    // It should load some items from the backend API, or "No conversation history" if empty
    // Just verifying it renders something inside
    await expect(memoryList.locator('.memory-item').first()).toBeVisible();
  });

  test('should execute a command via quick chip', async ({ page }) => {
    // Wait for the chip to be visible
    const calcChip = page.locator('.chip', { hasText: 'Math Calc' });
    await expect(calcChip).toBeVisible();

    // Click the chip, which triggers the command execution
    await calcChip.click();

    // Wait for response display to update
    const responseText = page.locator('#responseText');
    const intentMeta = page.locator('#intentMeta');
    
    // Wait for the intent to change from standby/routing to TOOLS
    await expect(intentMeta).toContainText('INTENT: TOOLS', { timeout: 10000 });
    
    // Check if the result contains the calculation
    await expect(responseText).toContainText('result', { ignoreCase: true });
  });

  test('should execute a text command via input box', async ({ page }) => {
    const input = page.locator('#cmdInput');
    await input.fill('what is the weather in Chennai?');
    
    // Press enter or click button
    await page.locator('.send-btn').click();

    // Wait for intent meta to change
    const intentMeta = page.locator('#intentMeta');
    await expect(intentMeta).toContainText('INTENT: WEATHER', { timeout: 10000 });
    
    // Wait for response text to contain weather info
    const responseText = page.locator('#responseText');
    await expect(responseText).toContainText('Chennai', { ignoreCase: true });
  });

});
