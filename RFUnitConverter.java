/*
 * RF Engineering Unit Converter
 * Author: Christopher Garcia
 * Date: 12/30/2025
 *
 * Purpose: This program provides a menu-driven tool for converting between
 * common RF and electrical engineering units. Engineers can quickly convert
 * dBm to Watts, Watts to dBm, between frequency units, and Voltage to dBV.
 * The program continues running until the user chooses to exit.
 */

import java.util.Scanner;  // Import Scanner class for reading user input

public class RFUnitConverter {

    public static void main(String[] args) {

        // Create a Scanner object to read input from the keyboard
        Scanner scanner = new Scanner(System.in);

        // This variable controls the main program loop - when false, program exits
        boolean running = true;

        // Display welcome message to the user
        System.out.println("=== RF Engineering Unit Converter ===");
        System.out.println("Convert between common RF/electrical units\n");

        // Main program loop - keeps running until user chooses to exit
        while (running) {

            // Display the menu of available conversion options
            System.out.println("Select a conversion:");
            System.out.println("[1] dBm to Watts");
            System.out.println("[2] Watts to dBm");
            System.out.println("[3] Frequency Conversion");
            System.out.println("[4] Voltage to dBV");
            System.out.println("[0] Exit");
            System.out.print("Enter choice: ");

            // Variable to store user's menu selection
            int choice;

            // Try to read the user's input as an integer
            // If they enter something that isn't a number, catch the error
            try {
                choice = scanner.nextInt();
            } catch (Exception e) {
                // Clear the invalid input from the scanner
                scanner.nextLine();
                // Inform user of the error and restart the loop
                System.out.println("Invalid input. Please enter a number 0-4.\n");
                continue;  // Skip to next iteration of the while loop
            }

            // Process the user's menu selection using if-else statements
            if (choice == 0) {
                // User chose to exit - set running to false to end the loop
                running = false;
                System.out.println("\nThank you for using RF Unit Converter. Goodbye!");

            } else if (choice == 1) {
                // Convert dBm to Watts
                // Formula: Watts = 10^(dBm/10) / 1000
                System.out.print("Enter power in dBm: ");
                double dBm = scanner.nextDouble();

                // Perform the conversion calculation
                double watts = Math.pow(10, dBm / 10) / 1000;

                // Display the result with formatting
                System.out.printf("%.2f dBm = %.6f Watts\n\n", dBm, watts);

            } else if (choice == 2) {
                // Convert Watts to dBm
                // Formula: dBm = 10 * log10(Watts * 1000)
                System.out.print("Enter power in Watts: ");
                double watts = scanner.nextDouble();

                // Validate that watts is positive (can't take log of zero or negative)
                if (watts <= 0) {
                    System.out.println("Error: Watts must be a positive value.\n");
                } else {
                    // Perform the conversion calculation
                    double dBm = 10 * Math.log10(watts * 1000);

                    // Display the result with formatting
                    System.out.printf("%.6f Watts = %.2f dBm\n\n", watts, dBm);
                }

            } else if (choice == 3) {
                // Call separate method to handle frequency conversions
                // This keeps the main method cleaner and more organized
                frequencyConversion(scanner);

            } else if (choice == 4) {
                // Convert Voltage to dBV (decibels relative to 1 Volt)
                // Formula: dBV = 20 * log10(Voltage)
                System.out.print("Enter voltage in Volts: ");
                double volts = scanner.nextDouble();

                // Validate that voltage is positive
                if (volts <= 0) {
                    System.out.println("Error: Voltage must be a positive value.\n");
                } else {
                    // Perform the conversion calculation
                    double dBV = 20 * Math.log10(volts);

                    // Display the result with formatting
                    System.out.printf("%.4f V = %.2f dBV\n\n", volts, dBV);
                }

            } else {
                // User entered a number outside the valid range (not 0-4)
                System.out.println("Invalid selection. Please enter 0-4.\n");
            }
        }

        // Close the scanner to release system resources
        scanner.close();
    }

    /*
     * frequencyConversion method
     * Purpose: Handles conversions between frequency units (Hz, kHz, MHz, GHz)
     * Parameter: scanner - the Scanner object for reading user input
     * This is a separate method to keep the code organized and modular
     */
    public static void frequencyConversion(Scanner scanner) {

        // Display submenu for frequency unit options
        System.out.println("\nFrequency Units:");
        System.out.println("[1] Hz  [2] kHz  [3] MHz  [4] GHz");

        // Get the source unit from user
        System.out.print("Convert FROM (1-4): ");
        int fromUnit = scanner.nextInt();

        // Get the target unit from user
        System.out.print("Convert TO (1-4): ");
        int toUnit = scanner.nextInt();

        // Validate both selections are in range
        if (fromUnit < 1 || fromUnit > 4 || toUnit < 1 || toUnit > 4) {
            System.out.println("Invalid unit selection.\n");
            return;  // Exit the method early
        }

        // Get the value to convert
        System.out.print("Enter frequency value: ");
        double value = scanner.nextDouble();

        // Array of multipliers to convert each unit to Hz (base unit)
        // Index 0 unused, 1=Hz, 2=kHz, 3=MHz, 4=GHz
        double[] toHz = {0, 1, 1000, 1000000, 1000000000};

        // Array of unit names for display purposes
        String[] unitNames = {"", "Hz", "kHz", "MHz", "GHz"};

        // Convert: first to Hz (multiply), then to target unit (divide)
        double inHz = value * toHz[fromUnit];
        double result = inHz / toHz[toUnit];

        // Display the conversion result
        System.out.printf("%.4f %s = %.4f %s\n\n",
                          value, unitNames[fromUnit],
                          result, unitNames[toUnit]);
    }
}
